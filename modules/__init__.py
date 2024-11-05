import os

try:
  import google.colab # type: ignore
  IN_COLAB = True
except:
  IN_COLAB = False
  
is_local = os.getenv("MYAPP_DEV_ENV") == "true"