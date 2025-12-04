# algorithmic-trading

## File structure
* `/data/raw` # csv files from yfinance (ignored by git)
* `/data/clean` # processed parquet files (ignored by git)
* `/notebooks` # shared notebooks for analysis
* `/reports` # strategy pdfs and final deliverables
* `/src` # main python package
    * `backtester.py` # execution engine and strategy interface
    * `clean_data.py` # script to generate clean data
    * `/strategies` # includes ma_crossover + mean_reversion
    * `/risk` # for metrics like sharpe 
* `/tests` # tests (to be implemented later)
* `.gitignore` # prevents data uploads
* `environment.yml` # conda environment setup

## Development practices

**1. How to Work**
* **Recommended:** Clone the repo locally. Always run `git pull` before working to get recent changes. When done, `git commit` and `git push`.
* **Otherwise (not recomended):** You can work separately and upload files manually to GitHub website, but this makes merging code difficult.

**2. Setup**
Run this command to make sure everyone has same dependencies:
`conda env create -f environment.yml`.
Then activate it:
`conda activate algo_trading`

**3. Documentation**
Remember to include comments in your code to make it understandable for others. Include a README.md in the directory if you think necessary.

**4. Data Files**
Data files are listed in `.gitignore`. Do not force commit them, because files over 100MB make the push fail .

## Contributors
To test that committing works, you can add your name below:
* Mea