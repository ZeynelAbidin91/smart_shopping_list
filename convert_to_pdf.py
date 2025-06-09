import markdown
import pdfkit
import os

# Convert markdown to HTML
with open('Development_Report.md', 'r', encoding='utf-8') as f:
    md_content = f.read()
    
html_content = markdown.markdown(md_content)

# Create a complete HTML document with some basic styling
html_document = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Development Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
        }}
        h1, h2, h3 {{ color: #333; }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
'''

# Write the HTML to a temporary file
with open('temp.html', 'w', encoding='utf-8') as f:
    f.write(html_document)

# Convert HTML to PDF
try:
    # Configure path to wkhtmltopdf
    path_wkthmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
    
    # Convert to PDF with configuration
    pdfkit.from_file('temp.html', 'Development_Report.pdf', 
                     configuration=config,
                     options={'enable-local-file-access': None})
    print("Successfully created Development_Report.pdf")
except Exception as e:
    print(f"Error creating PDF: {e}")
finally:
    # Clean up temporary HTML file
    if os.path.exists('temp.html'):
        os.remove('temp.html')
