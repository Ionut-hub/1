import requests
from bs4 import BeautifulSoup

def fetch_information(idee):
    # Exemplu: căută informații pe un site de știri sau pe un blog
    url = f"https://example.com/search?q={idee.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extragere informații relevante
    # Acest exemplu este simplificat și ar trebui adaptat în funcție de site-ul specific
    titluri = [h2.text for h2 in soup.find_all('h2')]
    rezultate = titluri[:5]  # Luați primele 5 rezultate
    return rezultate