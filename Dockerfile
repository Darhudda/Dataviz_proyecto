# Usa una imagen oficial de Python
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia todos los archivos al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde correrá la app
EXPOSE 8050

# Comando para correr la app (NO incluir cargar_datos.py aquí)
CMD ["python", "app.py"]

# Comando final: primero carga datos, luego corre la app
#MD python cargar_datos.py && python app.py
