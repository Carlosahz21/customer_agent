import pandas as pd
# Importar la función get_vector_store
from src.tools import get_vector_store  
from src.tools import search_products  
from src.tools import search_tool  
from src.tools import structured_search_tool
from src.tools import set_user_id

DEFAULT_USER_ID = 2
# Establecer el ID del usuario
set_user_id(DEFAULT_USER_ID)  # Puedes usar DEFAULT_USER_ID o cualquier ID válido de VALID_USER_IDS

# Realizar la búsqueda
results4 = structured_search_tool({
    "product_name": "Yogurt",
    "department": "dairy eggs",
    "top_k": 5,
    "order_by": "count",
    "ascending": True,
    "history_only": True
})

print(results4)