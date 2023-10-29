import requests
from bs4 import BeautifulSoup
from ipywidgets import IntProgress
from IPython.display import display
import os

def scrape_pdfs(conf_name, year, num_papers, root_path = 'pdfs'):
    # Create the url to scrape the pdfs
    url = f'https://aclanthology.org/events/{conf_name}-{year}/'

    # Get the Response variable
    response = requests.get(url)

    # Check if URL exists
    if response.status_code != 200:
        print(f"Error: Unable to access {url}")
        return

    # Instatiate the BeautifulSoup Object
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all hyperlinks present on webpage
    links = soup.find_all('a')

    # Check if links list is empty
    if not links:
        print("Error: No links found on the webpage.")
        return

    # Create directory structure if it doesn't exist
    dir_path = os.path.join(root_path, conf_name, year)
    os.makedirs(dir_path, exist_ok=True)

    # Initialize a counter to check if num_pages is reached
    i = 0

    # Display progress
    f = IntProgress(min=0, max=num_papers, description=f"{conf_name} PDFs...", style={'bar_color': 'green'}) # instantiate the bar
    display(f) # display the bar

    # From all links, check for pdf link of a paper and if present, download it
    for link in links:
        if ('.pdf' in link.get('href', [])) and ('Open PDF' == link.get('title', [])):
            i += 1
    
            # Get response object for link
            response = requests.get(link.get('href'))

            # Check if there was an error downloading the PDF
            if response.status_code != 200:
                print(f"Error: Unable to download file {i}")
                continue

            print("Downloading file: ", i)

            # Write content in pdf file

            file_path = os.path.join(dir_path, f"{conf_name}-{year}-{str(i)}.pdf")
            with open(file_path, 'wb') as pdf:
                pdf.write(response.content)
            pdf.close()
            f.value += 1 # signal to increment the progress bar
            if i>=num_papers:
                break
    f.bar_style = "success"
    f.close()
    print("Done.")