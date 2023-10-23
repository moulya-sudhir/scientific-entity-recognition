# end-to-end-NLP

## Link to Website: https://aclanthology.org/

### File Structure:
- <b>webscrape.py</b> : Contains function to scrape and download   pdfs from ACL Anthology website, based on conference name, year and number of pdfs
    - Example Usage: 
        - from webscrape import scrape_pdfs
        - scrape_pdfs('acl','2022', 5)
- <b>data folder</b> : folder containing all scientific paper data for all 3 conferences
    - Each folder has a <b>pdfs</b> folder, <b>tokens</b> folder, and an <b>annotations</b> folder.
- <b>test_webscrape.ipynb</b> : Examples of how to use the webscrape function
