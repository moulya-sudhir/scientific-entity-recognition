# end-to-end-NLP

## Link to Website: https://aclanthology.org/

### File Structure:
- <b>webscrape.py</b> : Contains function to scrape and download   pdfs from ACL Anthology website, based on conference name, year and number of pdfs
    - Example Usage: 
        - from webscrape import scrape_pdfs
        - scrape_pdfs('acl','2022', 5)
- <b>pdfs folder</b> : folder containing pdfs according to conference and year
- <b>test_webscrape.ipynb</b> : Examples of how to use the webscrape function
