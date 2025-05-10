# Usa una imagen oficial de Python
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos necesarios
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde correrá la app
EXPOSE 8050

# Comando para correr la app
CMD ["python", "app.py"]
