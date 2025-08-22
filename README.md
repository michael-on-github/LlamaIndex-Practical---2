# LlamaIndex-Practical---2


## Prerequisites

Access to Azure:

```bash
brew install azure-cli
az login
(select your subscription)
```


## Installation

```bash
tar xzf bge-small-en.tgz
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
(edit .env)
```

Create database container:

```bash
docker-compose up -d
```

Fill it with some data:

```bash
docker exec -it postgres_db psql -U postgres -d postgres

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO users (name, email) VALUES
('Alice Example', 'alice@example.com'),
('Bob Test', 'bob@test.com');

SELECT * FROM users;

\q
```

## Usage

```bash
streamlit run main.py
```

Now you can either load candidates' data via button in web UI and:

- ask specific questions about the loaded data:
   + "What were the working years of a candidate who was Logistics Coordinator?"
     – the answer should be around "10/2006 to 11/2007"
   + "What were the working years of a candidate who was Production Control Analyst?"
     – the answer should be around "11/2004 to 05/2006"
- ask questions about the database ("What tables does this database contain?" -- the answer should be "users") or
- run instructions against it ("Describe the first table", "Retrieve the first row of that table") or
- ask generic questions ("Can I eat rocks?")
