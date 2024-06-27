import qrcode
import streamlit as st
from PIL import Image
import io
import qrcode
import streamlit as st
from PIL import Image
import io

# Fonction pour générer le QR code
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    
    # Convertir l'image en bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    return byte_im

# URL de votre application Streamlit
app_url = "https://pr-diction-nbremoyenindice.streamlit.app"

# Génération du QR code
qr_image = generate_qr_code(app_url)

# Affichage du QR code dans Streamlit
st.image(qr_image, caption='Scannez pour accéder à l\'application', use_column_width=True)
