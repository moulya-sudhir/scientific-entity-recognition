# end-to-end-NLP

## Link to Website: https://aclanthology.org/

### File Structure:
- <b>code/webscrape.py</b> : Contains function to scrape and download   pdfs from ACL Anthology website, based on conference name, year and number of pdfs
    - Example Usage: 
        - from webscrape import scrape_pdfs
        - scrape_pdfs('acl','2022', 5)
- <b>code/runner.py</b> : Contains functions to train the model and test via a test set
    - Example Usage:
        - python code/runner.py --train --model_name 'roberta-large' --epochs 5 --batch_size 4 --lr 2e-5  --output_dir 'models/roberta-large' --train_data 'data/train' 
        - python code/runner.py --test --model_name 'models/roberta-large' --batch_size 4 --output_dir 'models/roberta-large' --test_data 'data/test'

- <b>code/prediction.py</b> : Contains code to predict on the Kaggle test sets
    - Example Usage:
        - python code/prediction.py --model_name 'models/roberta-large' --test_csv 'data/test.csv' --output_csv 'data/outputs.csv'

- <b>data folder</b> : folder containing all scientific paper data for all 3 conferences
    - Each folder has a <b>pdfs</b> folder, <b>tokens</b> folder, and an <b>annotations</b> folder.
- <b>test_webscrape.ipynb</b> : Examples of how to use the webscrape function
