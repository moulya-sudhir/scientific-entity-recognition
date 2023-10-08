# end-to-end-NLP

## Link to Website: https://aclanthology.org/

### File Structure:
- webscrape.py : Contains function to scrape and download   pdfs from ACL Anthology website, based on conference name, year and number of pdfs
    - Example Usage: 
        - from webscrape import scrape_pdfs
        - scrape_pdfs('acl','2022', 5)
- pdfs folder : folder containing pdfs according to conference and year
- test_webscrape.ipynb : Examples of how to use the webscrape function