## Data

```bash
mkdir data
```

### Hint

```bash
git clone https://github.com/futianfan/clinical-trial-outcome-prediction.git data/clinical-trial-outcome-prediction
```

### LLM output

upload prepared_data.zip to data

```bash
unzip data/prepared_data.zip
mv data/prepared_data/clinical-trial-outcome-prediction/* data/clinical-trial-outcome-prediction/data
mv data/prepared_data/clinical_trials_gov data
rm -rf data/prepared_data
```

## API key

#### Wandb

We use Wandb as our logger, so you have to create an account and login with your api key before running experiments.

```bash
wandb login
### paste your api key
```

#### OpenAI API key (optional)

We use OpenAI API to generate summaries, but we already provide all the data you need. You do not need to set this unless you want to generate summaries by yourself.

```bash
echo <your-openai-key> > openai_api_key.txt
```
