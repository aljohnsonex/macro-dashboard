# Macro Economic Data Dashboard

Interactive visualization of macroeconomic data from BigQuery using Streamlit.

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your BigQuery credentials. You have two options:

**Option 1: Full JSON (Recommended)**
```env
GCP_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}
```

**Option 2: Individual Fields**
```env
GCP_TYPE=service_account
GCP_PROJECT_ID=your-project-id
GCP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
GCP_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
# ... etc
```

Also set your BigQuery configuration:
```env
BQ_PROJECT_ID=your-project-id
BQ_DATASET_ID=your-dataset-id
BQ_TABLE_ID=macro_consolidated
```

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

## Security Note

The `.env` file is gitignored and will not be committed to version control. Never commit your actual credentials to the repository.

