# ==========================================
# generate_report.py (cu suport diacritice)
# ==========================================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os
import sys

# Calea completă către fontul DejaVuSans.ttf
font_path = os.path.join(os.path.dirname(__file__), "dejavu-fonts-ttf-2.37", "ttf", "DejaVuSans.ttf")

# Verificăm dacă fontul există
if not os.path.exists(font_path):
    print(f"❌ Fontul nu a fost găsit la: {font_path}")
    sys.exit(1)

# Înregistrăm fontul
pdfmetrics.registerFont(TTFont('DejaVu', font_path))

def create_report(accuracy):
    c = canvas.Canvas("Raport_Sarcina3.pdf", pagesize=A4)
    width, height = A4

    # Folosim fontul DejaVu pentru diacritice
    c.setFont("DejaVu", 14)
    c.drawString(2*cm, height-2*cm, "Sarcina 3 - Predicția categoriei produsului pe baza titlului")

    c.setFont("DejaVu", 11)
    c.drawString(2*cm, height-3*cm, f"Acuratețe model: {accuracy:.3f}")
    c.drawString(2*cm, height-4*cm, "Model: Logistic Regression + TF-IDF (1-2 n-gram)")
    c.drawString(2*cm, height-5*cm, "Set de date: IMLP4_TASK_03-products.csv")

    # Dacă există imaginea cu matricea de confuzie, o includem
    if os.path.exists("confusion_matrix.png"):
        c.drawImage("confusion_matrix.png", 2*cm, height-17*cm,
                    width=15*cm, preserveAspectRatio=True)
    else:
        c.drawString(2*cm, height-7*cm, "(Imaginea cu matricea de confuzie nu a fost găsită)")

    c.showPage()
    c.save()
    print("📄 Raport_Sarcina3.pdf a fost generat cu succes (cu diacritice).")

# Apelează funcția
create_report(0.953)
