# src/tools.py
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from functools import reduce

_current_user_id: Optional[int] = None


def set_user_id(uid: int):
    global _current_user_id
    _current_user_id = uid


def get_user_id() -> Optional[int]:
    return _current_user_id


# Global CSVs — loaded once, outside tool
PRODUCTS_CSV = "./dataset/products.csv"
DEPARTMENTS_CSV = "./dataset/departments.csv"

_products_df = pd.read_csv(PRODUCTS_CSV)
_product_lookup = dict(zip(_products_df["product_id"], _products_df["product_name"]))

# Load global CSVs once
products = pd.read_csv("./dataset/products.csv")
departments = pd.read_csv("./dataset/departments.csv")
aisles = pd.read_csv("./dataset/aisles.csv")
prior = pd.read_csv("./dataset/order_products__prior.csv")
orders = pd.read_csv("./dataset/orders.csv")


# Build enumerated options
DEPARTMENT_NAMES = sorted(departments["department"].dropna().unique().tolist())

VALID_USER_IDS = sorted(orders["user_id"].dropna().unique().tolist())

# Pick the first user for simplicity and safety
DEFAULT_USER_ID = VALID_USER_IDS[0]

# TODO
@tool
def structured_search_tool(
    product_name: Optional[str] = None,
    department: Optional[Literal[tuple(DEPARTMENT_NAMES)]] = None,
    aisle: Optional[str] = None,
    reordered: Optional[bool] = None,
    min_orders: Optional[int] = None,
    order_by: Optional[Literal["count", "add_to_cart_order"]] = None,
    ascending: Optional[bool] = False,
    top_k: Optional[int] = None,
    group_by: Optional[Literal["department", "aisle"]] = None,
    history_only: Optional[bool] = False,
) -> list:
    """
    A LangChain-compatible tool for structured product discovery across a grocery catalog and user purchase history.

    This function is decorated with `@tool` to expose it to an LLM agent via LangGraph. It supports SQL-like filtering
    over a product database, optionally constrained to the current user's order history. It returns either individual products
    or group-wise summaries based on the provided arguments.

    ---
    Tool Behavior Overview:
    - Operates on two global pandas DataFrames:
        • `products`: full catalog (from `products.csv`)
        • `prior` + `orders`: user order history (from `order_products__prior.csv`, `orders.csv`)
    - Uses additional joins with `departments.csv` and `aisles.csv` to enrich the product metadata.
    - If `history_only=True`, it will:
        • Look up the current user ID from `get_user_id()`
        • Merge product purchases for that user
        • Calculate statistics like reorder count, order frequency, and cart placement
    - Applies filters conditionally based on which arguments are set.
    - Optionally groups results by department or aisle.

    ---
    Parameters:
    - `product_name` (str, optional): Case-insensitive substring match on product names.
      Example: "almond" → matches "Almond Milk", "Almond Butter".

    - `department` (str, optional): Exact match against department names (from `DEPARTMENT_NAMES`).
      Example: "beverages", "pantry", "produce".

    - `aisle` (str, optional): Lowercased match on aisle name. Example: "organic snacks", "soy milk".

    - `reordered` (bool, optional): Only meaningful if `history_only=True`.
      - `True` → return only products reordered at least once
      - `False` → return products bought once, never reordered

    - `min_orders` (int, optional): Only meaningful if `history_only=True`.
      Filters for items purchased this many times or more.

    - `order_by` (str, optional): Only meaningful if `history_only=True`.
      - `"count"` → total times product was ordered
      - `"add_to_cart_order"` → average position in cart

    - `ascending` (bool, optional): Whether to sort `order_by` field ascending (default is `False` = descending).

    - `top_k` (int, optional): After filtering and sorting, returns only the top K products.

    - `group_by` (str, optional): If set to `"department"` or `"aisle"`, aggregates and returns counts instead of product rows.

    - `history_only` (bool, optional):
      - If `True`, only includes items the current user has purchased.
      - If `False` (default), searches the full catalog.

    ---
    Dependencies:
    - Requires global variables: `products`, `departments`, `aisles`, `prior`, `orders`
    - Requires user ID to be set via `set_user_id(user_id)` if `history_only=True`
    - Reads from CSVs under `./dataset/`

    ---
    Examples:
    ➤ Example 1: Find catalog items in pantry containing "peanut"
    ```json
    {
        "product_name": "peanut",
        "department": "pantry"
    }
    ```

    ➤ Example 2: Show reordered pantry products in my history
    ```json
    {
        "department": "pantry",
        "reordered": true,
        "history_only": true
    }
    ```

    ➤ Example 3: Top 5 most frequent purchases by user
    ```json
    {
        "order_by": "count",
        "top_k": 5,
        "history_only": true
    }
    ```

    ➤ Example 4: Count of catalog items by department
    ```json
    {
        "group_by": "department"
    }
    ```

    ---
    Returns:
    - If `group_by` is used:
      A list of dicts like:
      ```json
      [{"department": "pantry", "num_products": 132}, {"department": "beverages", "num_products": 89}]
      ```

    - Otherwise:
      A list of dicts, each with:
      ```json
      {
        "product_id": 24852,
        "product_name": "Organic Bananas",
        "aisle": "fresh fruits",
        "department": "produce",
        ... (optionally "count", "reordered", etc. if history_only=True)
      }
      ```

    - If no matches found, returns an empty list.
    - If required fields (e.g. user ID) are missing, returns a list with an error dict.

    ---
    LLM Usage Note:
    This tool is ideal for filtered browsing, purchase history analysis, or category breakdowns.
    """
    try:
        def clean_column(df, col):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype(int)
            return df

        def prepare_data():
            clean_column(products, "aisle_id")
            clean_column(products, "department_id")
            clean_column(aisles, "aisle_id")
            clean_column(departments, "department_id")

            catalog = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

            if not history_only:
                return catalog

            user_id = get_user_id()
            if user_id is None:
                raise ValueError("User ID not set. Use set_user_id() before calling this tool.")

            user_orders = orders[orders["user_id"] == user_id]
            user_prior = prior[prior["order_id"].isin(user_orders["order_id"])]
            enriched = user_prior.merge(products, on="product_id").merge(aisles, on="aisle_id").merge(departments, on="department_id")

            stats = enriched.groupby("product_id").agg(
                count=("order_id", "count"),
                add_to_cart_order=("add_to_cart_order", "mean"),
                reordered=("reordered", "max"),
            ).reset_index()

            enriched = enriched.drop_duplicates("product_id").merge(stats, on="product_id")
            return enriched

        def apply_filters(df):
            filters = [
                lambda d: d[d["product_name"].str.contains(product_name, case=False, na=False)] if product_name else d,
                lambda d: d[d["department"] == department] if department else d,
                lambda d: d[d["aisle"].str.contains(aisle, case=False, na=False)] if aisle else d,
                lambda d: d[d["reordered"] == reordered] if history_only and reordered is not None else d,
                lambda d: d[d["count"] >= min_orders] if history_only and min_orders is not None else d,
            ]
            return reduce(lambda acc, f: f(acc), filters, df)

        def sort_and_limit(df):
            if history_only and order_by:
                df = df.sort_values(by=order_by, ascending=ascending)
            return df.head(top_k) if top_k else df

        def format_output(df):
            if group_by:
                grouped = df.groupby(group_by).size().reset_index(name="num_products")
                return grouped.to_dict(orient="records")

            base_cols = ["product_id", "product_name", "aisle", "department"]
            extra_cols = [col for col in ["count", "reordered", "add_to_cart_order"] if col in df.columns]
            return df[base_cols + extra_cols].to_dict(orient="records")

        df = prepare_data()
        df = apply_filters(df)
        df = sort_and_limit(df)
        return format_output(df)

    except Exception as e:
        return [{"error": f"Error during structured search: {str(e)}"}]

# TODO
class RouteToCustomerSupport(BaseModel):
    """
    Pydantic schema for the assistant tool that triggers routing to customer support.

    This tool is used by the assistant to signal that the user has a problem beyond
    the scope of sales, such as refund requests or broken products.

    ---
    Fields:
    - reason (str): A short, human-readable message stating why support is needed.
      This must match the user's stated concern.

    ---
    Usage Requirements:
    - The assistant must populate this tool with the user's reason verbatim.
    - It must be called in tool_calls from the LLM when escalation is needed.
    - This tool is detected by `after_sales_tool(...)` to drive state transitions.

    ---
    Example:
    ```json
    {
        "reason": "My laptop arrived broken and I want a refund"
    }
    ```

    This schema must be registered as a tool in the assistant's tool list.
    """

    reason: str = Field(description="The reason why the customer needs support.")


class EscalateToHuman(BaseModel):
    severity: str = Field(
        description="The severity level of the issue (low, medium, high)."
    )
    summary: str = Field(description="A brief summary of the customer's issue.")


# ---- NEW: Search tool and handler ----
class Search(BaseModel):
    query: str = Field(description="User's natural language product search query.")


CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"
_embeddings = None
_vector_store = None


def get_vector_store():
    """
    Initialize and return the Chroma vector store instance.

    This function ensures that the vector store is loaded only once and reused across calls.
    It uses HuggingFace embeddings and connects to the Chroma database directory and collection.

    ---
    Returns:
    - Chroma: A Chroma vector store instance ready for similarity search.
    """
    global _vector_store, _embeddings

    if _vector_store is None:
        # Initialize embeddings
        if _embeddings is None:
            _embeddings = HuggingFaceEmbeddings(
                model_name="mixedbread-ai/mxbai-embed-large-v1",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        # Connect to Chroma vector store
        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )

    return _vector_store

def make_query_prompt(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query.strip().replace(chr(10), ' ')}"


# TODO
def search_products(query: str, top_k: int = 5):
    """
    Perform a semantic vector search over the product catalog using HuggingFace embeddings and Chroma.

    This function enables retrieval-augmented generation (RAG) by embedding a user's query and searching
    for the most relevant product entries using vector similarity.

    ---
    Requirements:
    - You must call `make_query_prompt(query: str) -> str` to wrap the query text.
    - You must call `get_vector_store()` to obtain a Chroma instance.
    - You must perform `similarity_search(query_text: str, k: int)` on the Chroma vector store.
    - Each result is a `Document` with `metadata` and `page_content`.

    ---
    Arguments:
    - query (str): A user query like "I want healthy snacks" or "almond milk".
    - top_k (int): Number of similar results to return (default: 5).

    ---
    Returns:
    - A list of dicts, each with the following fields:
        - "product_id" (int)
        - "product_name" (str)
        - "aisle" (str)
        - "department" (str)
        - "text" (str): full `page_content` of the result

    ---
    Example return:
    ```python
    [
        {
            "product_id": 123,
            "product_name": "Organic Almond Milk",
            "aisle": "Dairy Alternatives",
            "department": "Beverages",
            "text": "Organic Almond Milk, found in the Dairy Alternatives aisle..."
        }
    ]
    ```
    """
    # Obtener la base de datos vectorial
    vector_store = get_vector_store()
    # Generar el prompt para la consulta
    query_prompt = make_query_prompt(query)
    # Realizar la búsqueda de similitud
    results = vector_store.similarity_search(query_prompt, k=top_k)

    # Formatear los resultados
    return [
        {
            "product_id": doc.metadata["product_id"],
            "product_name": doc.metadata["product_name"],
            "aisle": doc.metadata["aisle"],
            "department": doc.metadata["department"],
            "text": doc.page_content,
        }
        for doc in results
    ]


# TODO
@tool
def search_tool(query: str) -> str:
    """
    Tool-decorated function that performs semantic product search using vector similarity,
    formats the results into a human-readable response, and is callable by a LangChain agent.

    This function is registered as a LangChain tool using the `@tool` decorator. It is intended
    for natural language queries from users looking for relevant products. Internally, it wraps
    `search_products(...)`, which uses a sentence embedding model and vector database.

    ---
    Tool Decorator:
    - This function is wrapped with `@tool` so that it can be invoked by an LLM agent during
      LangGraph execution when choosing from available tools.

    ---
    Arguments:
    - query (str): A free-form natural language string describing what the user is looking for.
      Examples:
        - "high protein vegan snacks"
        - "easy breakfast foods"
        - "organic almond butter"

    ---
    Internal Behavior:
    - Calls `search_products(query: str)` to perform semantic vector search.
    - The `search_products()` function:
        • Uses `make_query_prompt()` to convert the query into a format suitable for embedding.
        • Embeds the prompt using a HuggingFace sentence transformer.
        • Calls `get_vector_store()` to get a Chroma DB.
        • Returns metadata-rich matches including ID, name, aisle, department, and description.
    - The results (if any) are converted into a multi-line formatted string showing:
        - Product name and ID
        - Aisle and Department
        - Text description from vector DB

    ---
    Returns:
    - If products found:
        A formatted multiline string:
        ```
        - Organic Granola (ID: 18872)
          Aisle: Cereal
          Department: Breakfast
          Details: Organic Granola, found in the Cereal aisle...
        ```
    - If no products found:
        `"No products found matching your search."`

    ---
    Example Use (from an LLM):
    ```python
    search_tool("something high protein for breakfast")
    ```
    """
    results = search_products(query)
    if not results:
        return "No products found matching your search."
    # Format results into a readable string
    formatted_results = "\n".join(
        map(
            lambda item: (
                f"- {item['product_name']} (ID: {item['product_id']})\n"
                f"  Aisle: {item['aisle']}\n"
                f"  Department: {item['department']}\n"
                f"  Details: {item['text']}"
            ),
            results
        )
    )
    return formatted_results


# ---- UPDATED: Cart tools with quantity support ----
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Simulated in-memory cart storage - now stores product_id: quantity pairs
_cart_storage: Dict[str, Dict[int, int]] = {}
_current_thread_id: Optional[str] = None


def set_thread_id(tid: str):
    global _current_thread_id
    _current_thread_id = tid


def get_cart() -> Union[List[str], Dict[int, int]]:
    if _current_thread_id is None:
        return ["Session error: no thread ID set."]
    return _cart_storage.setdefault(_current_thread_id, {})


@tool
def cart_tool(
    cart_operation: str, product_id: Optional[int] = None, quantity: int = 1
) -> str:
    """
    Modify the user's cart by adding, removing, updating quantity, or buying products.

    Args:
        cart_operation: The operation to perform (add, remove, update, buy)
        product_id: The ID of the product to add, remove, or update
        quantity: The quantity for add or update operations (default: 1)
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if cart_operation == "add":
        if product_id is None:
            return "No product ID provided to add."

        # If product already exists, increase quantity
        if product_id in cart:
            cart[product_id] += quantity
            return f"Added {quantity} more of product {product_id} to your cart. New quantity: {cart[product_id]}."
        else:
            cart[product_id] = quantity
            product_name = _product_lookup.get(product_id, "Unknown Product")
            return (
                f"Added {quantity} of {product_name} (ID: {product_id}) to your cart."
            )

    elif cart_operation == "update":
        if product_id is None:
            return "No product ID provided to update."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        cart[product_id] = quantity
        product_name = _product_lookup.get(product_id, "Unknown Product")
        return f"Updated quantity of {product_name} (ID: {product_id}) to {quantity}."

    elif cart_operation == "remove":
        if product_id is None:
            return "No product ID provided to remove."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        product_name = _product_lookup.get(product_id, "Unknown Product")

        # If quantity is specified and less than current quantity, reduce quantity
        if quantity > 1 and cart[product_id] > quantity:
            cart[product_id] -= quantity
            return f"Removed {quantity} of {product_name} (ID: {product_id}) from your cart. New quantity: {cart[product_id]}."
        else:
            # Otherwise remove product completely
            del cart[product_id]
            return f"Removed {product_name} (ID: {product_id}) from your cart."

    elif cart_operation == "buy":
        if not cart:
            return "Your cart is empty. Nothing to purchase."
        cart.clear()
        return "Thank you for your purchase! Your cart is now empty."

    return f"Unknown cart operation: {cart_operation}"


@tool
def view_cart() -> str:
    """
    Display the contents of the user's cart with quantities.
    This is a standard tool that returns a formatted string representation of the cart.
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if not cart:
        return "Your cart is currently empty."

    lines = ["Your cart contains:"]
    for pid, qty in cart.items():
        title = _product_lookup.get(pid, "Unknown Product")
        lines.append(f"- {title} (ID: {pid}) × {qty}")
    return "\n".join(lines)


# ---- Tool fallback handling ----


def handle_tool_error(state: Dict[str, Any]) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistake.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# TODO
def create_tool_node_with_fallback(tools: list) -> ToolNode:
    """
    Build a LangGraph ToolNode that can handle errors gracefully using a fallback strategy.

    This function should create a ToolNode that wraps a list of tools and attaches
    a fallback mechanism using LangChain's `with_fallbacks(...)` method.

    ---
    Requirements:
    - Return a `ToolNode` from `langgraph.prebuilt`.
    - Attach a fallback using `.with_fallbacks(...)` with your error handler.
    - Use `handle_tool_error(state)` as the fallback function.
    - Set `exception_key="error"` so LangGraph recognizes failure states.

    ---
    Arguments:
    - tools (list): A list of @tool-decorated functions (LangChain tools).

    ---
    Returns:
    - ToolNode: A LangGraph-compatible tool node with error fallback logic.
    """
    return ToolNode(tools).with_fallbacks(
    [RunnableLambda(handle_tool_error)],
    exception_key="error"
)


__all__ = [
    "RouteToCustomerSupport",
    "EscalateToHuman",
    "Search",
    "search_products",
    "search_tool",
    "cart_tool",
    "view_cart",
    "handle_tool_error",
    "create_tool_node_with_fallback",
    "structured_search_tool",
]
