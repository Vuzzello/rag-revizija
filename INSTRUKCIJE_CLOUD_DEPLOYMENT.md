# RAG Interna Revizija — Cloud Deployment Vodič
## Streamlit Cloud + Supabase + Groq + HuggingFace

---

## Pregled arhitekture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KORISNIK (Browser)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│          STREAMLIT COMMUNITY CLOUD (besplatno)                   │
│          https://your-app.streamlit.app                          │
└──────┬────────────────────┬──────────────────┬──────────────────┘
       │                    │                  │
┌──────▼──────┐  ┌──────────▼──────┐  ┌───────▼────────────────┐
│  SUPABASE   │  │   GROQ API      │  │  HUGGINGFACE API        │
│  pgvector   │  │   (LLM)         │  │  (Embeddings)           │
│  (baza)     │  │   Llama 3.3 70B │  │  BAAI/bge-m3            │
│  besplatno  │  │   besplatno     │  │  besplatno              │
└─────────────┘  └─────────────────┘  └────────────────────────┘
```

## Servisi koji su potrebni (svi besplatni):

| Servis | Svrha | Limit besplatnog plana |
|--------|-------|----------------------|
| Streamlit Community Cloud | Hosting app | Neograničeno (javne app) |
| Supabase | Vector baza podataka | 500MB, 2 projekta |
| Groq | LLM generacija | 14.400 req/dan, 30 req/min |
| HuggingFace | Embedding model API | 1000 req/dan |
| GitHub | Source code | Neograničeno |

---

## KORAK 1 — GitHub repozitorij

### 1.1 Kreiranje GitHub naloga
1. Idi na https://github.com
2. Klikni **Sign up** — registruj se sa email adresom
3. Potvrdi email

### 1.2 Kreiranje repozitorija
1. Nakon prijave, klikni **"New"** (zeleno dugme) ili idi na https://github.com/new
2. Popuni:
   - **Repository name:** `rag-revizija`
   - **Description:** `RAG baza znanja za interne revizore`
   - Odaberi: **Private** (preporučeno za interne alate)
   - ✅ Čekiraj **"Add a README file"**
3. Klikni **"Create repository"**

### 1.3 Instalacija Git-a (ako nemaš)
1. Preuzmi sa: https://git-scm.com/download/win
2. Instaliraj sa defaultnim opcijama
3. Provjera: otvori CMD i ukucaj `git --version`

### 1.4 Upload projekta na GitHub
Otvori Command Prompt u folderu projekta:

```cmd
cd C:\Projekti\rag_revizija

git init
git add .
git commit -m "Inicijalni commit — RAG Interna Revizija"
git branch -M main
git remote add origin https://github.com/TVOJE_IME/rag-revizija.git
git push -u origin main
```

> ⚠️ VAŽNO: Nikada ne stavljaj `.env` fajl na GitHub!
> Provjeri da `.gitignore` sadrži `.env` (već je uključen u projektu)

---

## KORAK 2 — Supabase (Vector baza podataka)

### 2.1 Kreiranje naloga
1. Idi na: https://supabase.com
2. Klikni **"Start your project"**
3. Registruj se sa GitHub nalogom (preporučeno) ili email-om

### 2.2 Kreiranje projekta
1. Klikni **"New project"**
2. Popuni:
   - **Organization:** tvoje ime (defaultno)
   - **Name:** `rag-revizija`
   - **Database Password:** kreiraj jaku lozinku i **SAČUVAJ JE NEGDJE SIGURNO**
   - **Region:** odaberi `EU West` (Frankfurt) — najbliže Balkanu
3. Klikni **"Create new project"**
4. Čekaj ~2 minute dok se projekt kreira

### 2.3 Dohvati API ključeve
1. U lijevom meniju klikni **Settings** (zupčanik ikona)
2. Klikni **API**
3. Kopiraj i sačuvaj:
   - **Project URL** → ovo je tvoj `SUPABASE_URL`
   - **anon / public key** → ovo je tvoj `SUPABASE_KEY`

### 2.4 Pokreni SQL setup
1. U lijevom meniju klikni **SQL Editor**
2. Klikni **"New query"**
3. Kopiraj CIJELI sadržaj fajla `sql/01_setup.sql` iz projekta
4. Zalijepi u editor
5. Klikni **"Run"** (ili `Ctrl+Enter`)
6. Trebalo bi da vidiš: `Success. No rows returned`

### 2.5 Provjera
1. U lijevom meniju klikni **Table Editor**
2. Trebalo bi da vidiš tabelu `dokumenti`
3. Klikni na nju — trebalo bi da je prazna (normalno)

---

## KORAK 3 — Groq API (LLM)

### 3.1 Kreiranje naloga
1. Idi na: https://console.groq.com
2. Klikni **"Sign up"**
3. Registruj se (Google ili email)
4. Potvrdi email

### 3.2 Kreiranje API ključa
1. U lijevom meniju klikni **"API Keys"**
2. Klikni **"Create API Key"**
3. **Name:** `rag-revizija`
4. Klikni **"Submit"**
5. **ODMAH KOPIRAJ KLJUČ** — prikazuje se samo jednom!
6. Sačuvaj kao `GROQ_API_KEY`

### 3.3 Odabrani model
Koristimo: **`llama-3.3-70b-versatile`**
- Veoma kvalitetan za višejezične tekstove
- Besplatan: 14.400 zahtjeva/dan, 30 zahtjeva/minuti
- Alternativa za brže odgovore: `llama-3.1-8b-instant`

---

## KORAK 4 — HuggingFace API (Embeddings)

### 4.1 Kreiranje naloga
1. Idi na: https://huggingface.co
2. Klikni **"Sign Up"**
3. Registruj se

### 4.2 Kreiranje API tokena
1. Klikni na svoju profilnu sliku (gore desno)
2. Odaberi **"Settings"**
3. U lijevom meniju klikni **"Access Tokens"**
4. Klikni **"New token"**
5. Popuni:
   - **Name:** `rag-revizija`
   - **Role:** `read`
6. Klikni **"Generate a token"**
7. **KOPIRAJ TOKEN** — sačuvaj kao `HF_API_TOKEN`

### 4.3 Provjera modela
Model koji koristimo (`BAAI/bge-m3`) je javno dostupan.
Provjeri na: https://huggingface.co/BAAI/bge-m3

---

## KORAK 5 — Streamlit Community Cloud (Hosting)

### 5.1 Kreiranje naloga
1. Idi na: https://streamlit.io/cloud
2. Klikni **"Sign up"**
3. **Obavezno:** Registruj se sa ISTIM GitHub nalogom koji si koristio za repozitorij

### 5.2 Deploy aplikacije
1. Nakon prijave, klikni **"New app"**
2. Popuni:
   - **Repository:** `TVOJE_IME/rag-revizija`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **App URL:** `rag-revizija` (ili po želji)
3. Klikni **"Advanced settings"**

### 5.3 Dodavanje tajnih varijabli (SECRETS)
U **"Advanced settings"** → sekcija **"Secrets"**:

Zalijepi sljedeće (zamijeni vrijednosti sa svojim ključevima):

```toml
SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
RETRIEVAL_TOP_K = "5"
RETRIEVAL_SCORE_THRESHOLD = "0.35"
```

4. Klikni **"Deploy!"**
5. Čekaj 3-5 minuta dok se app builda

### 5.4 Provjera deploymenta
- Otvori URL koji ti je dat (npr. `https://rag-revizija.streamlit.app`)
- Trebalo bi da vidiš aplikaciju
- Status indikatori trebaju biti zeleni

---

## KORAK 6 — Widget za ugradnju na website

### 6.1 Jednostavan iframe widget
Kopiraj kod iz fajla `widget/widget.html` i ugradi ga na svoju stranicu.

### 6.2 Ugradnja u WordPress
1. Dodaj novi Page ili Post
2. Dodaj **HTML blok**
3. Zalijepi iframe kod

### 6.3 Ugradnja u bilo koji website
```html
<iframe
  src="https://rag-revizija.streamlit.app/?embed=true"
  width="100%"
  height="800px"
  frameborder="0"
  style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);"
></iframe>
```

---

## Ažuriranje aplikacije (nakon izmjena koda)

```cmd
cd C:\Projekti\rag_revizija
git add .
git commit -m "Opis izmjene"
git push
```

Streamlit Community Cloud automatski detektuje push i redeploya app (~2 minute).

---

## Troškovi

Sve je **besplatno** dok god:
- Supabase baza < 500MB (dovoljno za ~50.000 chunkova)
- Groq < 14.400 req/dan
- HuggingFace < 1000 embedding req/dan
- Streamlit: uvijek besplatno za javne app

Za veće potrebe:
- Supabase Pro: $25/mj (8GB)
- Groq Pay-as-you-go: ~$0.59/1M tokena
- HuggingFace PRO: $9/mj (neograničeni API pozivi)

---

*Dokument verzija 1.0 | Ažurirano: 2025-04-27*
