from fpdf import FPDF

# create FPDF object
pdf = FPDF()

# add a page
pdf.add_page()

# set font and font size
pdf.set_font("Arial", size=12)

# add text
pdf.cell(200, 10, "My Analysis Report", ln=1, align='C')

# add a table
data = [['Name', 'Age', 'Gender'],
        ['John', '25', 'Male'],
        ['Jane', '30', 'Female'],
        ['Bob', '40', 'Male']]
for row in data:
    for item in row:
        pdf.cell(50, 10, str(item), border=1)
    pdf.ln()

# add a mathematical expression
pdf.cell(200, 10, "The quadratic formula is: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$", ln=1)

# render an image
pdf.cell(200, 10, "My Plot 1", ln=1)
pdf.image("dendrogram-single.png", x=None, y=None, w=0, h=0, type='', link='')

# add some more text
pdf.cell(200, 10, "Some more text goes here.", ln=1)

# save the PDF file
pdf.output("my_analysis_report.pdf")
