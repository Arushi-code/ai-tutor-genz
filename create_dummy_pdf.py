from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_dummy_textbook(filename="state_board_textbook.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Chapter 1
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "Chapter 1: The Solar System")
    c.setFont("Helvetica", 12)
    text1 = (
        "The Solar System consists of the Sun and the planetary system that orbits it. "
        "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, "
        "Saturn, Uranus, and Neptune. The Earth is the third planet from the Sun and the only "
        "known planet to support life. Jupiter is the largest planet, famous for its Great Red Spot."
    )
    # Basic text wrapping
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(text1, "Helvetica", 12, width - 144)
    y = height - 100
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
        
    c.showPage()
    
    # Chapter 2
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "Chapter 2: History of India")
    c.setFont("Helvetica", 12)
    text2 = (
        "India has a rich and diverse history. The Indus Valley Civilization, one of the oldest "
        "in the world, flourished in the northwestern part of the Indian subcontinent from 3300 BCE "
        "to 1300 BCE. Fast forward to 1947, India gained independence from British rule after a long "
        "and arduous struggle led by figures like Mahatma Gandhi, Jawaharlal Nehru, and Subhas Chandra Bose."
    )
    lines = simpleSplit(text2, "Helvetica", 12, width - 144)
    y = height - 100
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
        
    c.save()
    print(f"Created a sample PDF file: {filename}")

if __name__ == "__main__":
    create_dummy_textbook()
