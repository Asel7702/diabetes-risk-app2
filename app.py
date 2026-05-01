import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

import os
import requests
import gdown

@st.cache_resource
def load_model():
    model_path = "diabetes_model.pkl"

    if not os.path.exists(model_path):
        file_id = "10GJ21ni9XD3qSWtPaEztPgPuyra0293Q"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False, fuzzy=True)

    return joblib.load(model_path)

model = load_model()
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🧪",
    layout="wide"
)

# =========================
# OPENAI CLIENT
# =========================
from openai import OpenAI

import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# CSS — MEDICAL UI
# =========================
st.markdown("""
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes softPulse {
    0% { box-shadow: 0 0 0 rgba(14, 165, 233, 0.0); }
    50% { box-shadow: 0 14px 35px rgba(14, 165, 233, 0.18); }
    100% { box-shadow: 0 0 0 rgba(14, 165, 233, 0.0); }
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(45, 212, 191, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.20), transparent 32%),
        linear-gradient(135deg, #f8fafc 0%, #ecfeff 45%, #f0fdf4 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    border-radius: 30px;
    padding: 36px 32px;
    margin-bottom: 26px;
    background: linear-gradient(120deg, #0f766e, #0284c7, #22c55e);
    background-size: 250% 250%;
    animation: gradientMove 7s ease infinite, fadeUp 0.7s ease;
    color: white;
    box-shadow: 0 20px 45px rgba(15, 118, 110, 0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 850;
    margin-bottom: 10px;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.95;
}

.card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(186, 230, 253, 0.75);
    border-radius: 26px;
    padding: 26px;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.10);
    animation: fadeUp 0.75s ease;
    margin-bottom: 18px;
}

.card-glow {
    animation: fadeUp 0.75s ease, softPulse 3.8s ease-in-out infinite;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #0f766e;
    margin-bottom: 14px;
}

.helper {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 12px;
}

.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #0f766e !important;
    font-weight: 700 !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 16px;
    color: white;
    font-size: 16px;
    font-weight: 800;
    background: linear-gradient(135deg, #0f766e 0%, #0284c7 50%, #22c55e 100%);
    background-size: 220% 220%;
    animation: gradientMove 5s ease infinite;
    box-shadow: 0 12px 26px rgba(14, 165, 233, 0.25);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    color: white;
    box-shadow: 0 16px 32px rgba(14, 165, 233, 0.35);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(240,253,250,0.95), rgba(239,246,255,0.95));
    border: 1px solid rgba(125, 211, 252, 0.45);
    padding: 18px;
    border-radius: 18px;
}

[data-testid="stMetricLabel"] {
    color: #0f766e;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: #0f172a;
    font-weight: 850;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0f766e 0%, #0284c7 50%, #22c55e 100%);
    border-radius: 999px;
}

.explain-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.92));
    border: 1px solid rgba(125, 211, 252, 0.6);
    border-radius: 20px;
    padding: 20px;
    margin-top: 18px;
}

.explain-title {
    color: #0f766e;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 12px;
    color: white;
    background: linear-gradient(135deg, #0f766e 0%, #0284c7 100%);
}

.footer-note {
    color: #64748b;
    font-size: 13px;
    margin-top: 10px;
}

.stChatMessage {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(186, 230, 253, 0.6);
    border-radius: 18px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">🧪Diabetes Risk Assessment</div>
    <div class="hero-subtitle">
        AI-powered medical screening system for diabetes risk prediction
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================
def build_input_dataframe(model, gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    feature_names = list(model.feature_names_in_)
    input_dict = {col: 0 for col in feature_names}

    gender_num = 0 if gender == "Женский" else 1
    hypertension_num = 0 if hypertension == "Нет" else 1
    heart_num = 0 if heart_disease == "Нет" else 1

    base_values = {
        "gender": gender_num,
        "гендер": gender_num,
        "age": age,
        "возраст": age,
        "hypertension": hypertension_num,
        "гипертония": hypertension_num,
        "heart_disease": heart_num,
        "bmi": bmi,
        "BMI": bmi,
        "ИМТ": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }

    for col, value in base_values.items():
        if col in input_dict:
            input_dict[col] = value

    smoking_map = {
        "Никогда": "never",
        "В прошлом": "former",
        "Настоящее время": "current",
        "Нет информации": "no_info"
    }

    smoking_value = smoking_map[smoking_history]

    if "smoking_history" in input_dict:
        smoking_numeric = {
            "never": 4,
            "former": 3,
            "current": 1,
            "no_info": 0
        }
        input_dict["smoking_history"] = smoking_numeric[smoking_value]

    for col in feature_names:
        if col.lower().startswith("smoking_history_"):
            input_dict[col] = 0

    smoking_columns = {
        "current": ["smoking_history_current"],
        "former": ["smoking_history_former"],
        "never": ["smoking_history_never"],
        "no_info": ["smoking_history_no info", "smoking_history_no_info"]
    }

    for target_col in smoking_columns.get(smoking_value, []):
        for real_col in feature_names:
            if real_col.lower() == target_col.lower():
                input_dict[real_col] = 1

    return pd.DataFrame([input_dict], columns=feature_names)


def get_ai_response(user_message, result):
    chat_history = st.session_state.messages[-6:]

    system_prompt = """
Ты AI-ассистент в образовательном ML-проекте по оценке риска диабета.

Правила:
- отвечай на русском языке
- объясняй простыми словами
- не ставь диагноз
- не назначай лечение
- не пугай пользователя
- объясняй, какие показатели повлияли на риск
- давай мягкие рекомендации: питание, активность, контроль анализов, консультация врача
- всегда напоминай, что результат модели не заменяет врача
"""

    context = f"""
Результат модели:
Риск диабета: {result['risk'] * 100:.2f}%
Категория риска: {result['risk_text']}

Данные пациента:
Пол: {result['gender']}
Возраст: {result['age']}
Гипертония: {result['hypertension']}
Болезни сердца: {result['heart_disease']}
Курение: {result['smoking_history']}
ИМТ: {result['bmi']}
HbA1c: {result['hba1c']}
Глюкоза: {result['glucose']}

Факторы, повышающие риск:
{', '.join(result['risk_factors']) if result['risk_factors'] else 'не выявлены'}

Факторы, снижающие риск:
{', '.join(result['protective_factors']) if result['protective_factors'] else 'не выявлены'}

Вопрос пользователя:
{user_message}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": context})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.4
    )

    return response.choices[0].message.content


# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋Ввод данных пациента</div>', unsafe_allow_html=True)
    st.markdown('<div class="helper">Заполни данные и нажми кнопку расчёта</div>', unsafe_allow_html=True)

    gender = st.selectbox("Пол", ["Женский", "Мужской"])
    age = st.slider("Возраст", 1, 100, 25)
    hypertension = st.selectbox("Гипертония", ["Нет", "Да"])
    heart_disease = st.selectbox("Болезни сердца", ["Нет", "Да"])
    smoking_history = st.selectbox(
        "История курения",
        ["Никогда", "В прошлом", "Настоящее время", "Нет информации"]
    )

    bmi = st.number_input("ИМТ", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.2, step=0.1)
    glucose = st.number_input("Глюкоза", min_value=50, max_value=300, value=95, step=1)

    predict = st.button("Рассчитать риск")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card card-glow">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Результат</div>', unsafe_allow_html=True)

    if predict:
        try:
            input_data = build_input_dataframe(
                model=model,
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                smoking_history=smoking_history,
                bmi=bmi,
                hba1c=hba1c,
                glucose=glucose
            )

            risk = model.predict_proba(input_data)[0][1]

            st.metric("Риск диабета", f"{risk * 100:.2f}%")
            st.progress(float(risk))

            if risk < 0.30:
                st.success("✅ Низкий риск диабета")
                risk_text = "низкий"
                badge_text = "Низкий риск"
            elif risk < 0.70:
                st.warning("⚠️ Средний риск диабета")
                risk_text = "средний"
                badge_text = "Средний риск"
            else:
                st.error("🚨 Высокий риск диабета")
                risk_text = "высокий"
                badge_text = "Высокий риск"

            risk_factors = []
            protective_factors = []

            if age >= 45:
                risk_factors.append("возраст 45+")
            else:
                protective_factors.append("молодой возраст")

            if hypertension == "Да":
                risk_factors.append("наличие гипертонии")
            else:
                protective_factors.append("отсутствие гипертонии")

            if heart_disease == "Да":
                risk_factors.append("наличие сердечных заболеваний")
            else:
                protective_factors.append("отсутствие сердечных заболеваний")

            if bmi >= 30:
                risk_factors.append("высокий ИМТ")
            elif bmi < 25:
                protective_factors.append("нормальный ИМТ")

            if hba1c >= 6.5:
                risk_factors.append("повышенный HbA1c")
            else:
                protective_factors.append("нормальный HbA1c")

            if glucose >= 140:
                risk_factors.append("повышенный уровень глюкозы")
            else:
                protective_factors.append("нормальный уровень глюкозы")

            if smoking_history in ["В прошлом", "Настоящее время"]:
                risk_factors.append("история курения")
            else:
                protective_factors.append("отсутствие истории курения")

            st.session_state.last_result = {
                "risk": risk,
                "risk_text": risk_text,
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "hba1c": hba1c,
                "glucose": glucose,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors
            }

            st.markdown(
                f"""
                <div class="explain-box">
                    <div class="badge">{badge_text}</div>
                    <div class="explain-title"> Краткое объяснение</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"Модель оценивает риск диабета как **{risk_text}**.")

            if risk_factors:
                st.write("**Факторы, которые повышают риск:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")

            if protective_factors:
                st.write("**Факторы, которые снижают риск:**")
                for factor in protective_factors:
                    st.write(f"• {factor}")

            st.info("Это не медицинский диагноз. Результат модели не заменяет консультацию врача.")

        except Exception as e:
            st.error("Ошибка при предсказании.")
            st.write("Проверь, что файл diabetes_model.pkl находится рядом с app.py")
            st.code(str(e))

    else:
        st.markdown("""
        <div class="explain-box">
            <div class="badge">Ожидание</div>
            <div class="explain-title">Здесь появится результат</div>
            <div class="footer-note">
                Заполни данные слева и нажми кнопку расчёта риска
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💬 AI Chat")

    if st.session_state.last_result is None:
        st.info("Сначала рассчитай риск, потом AI-чат сможет объяснить результат.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Спроси про результат...")

        if user_prompt:
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt
            })

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("AI анализирует результат..."):
                    try:
                        assistant_reply = get_ai_response(
                            user_message=user_prompt,
                            result=st.session_state.last_result
                        )
                    except Exception as e:
                        assistant_reply = f"Ошибка AI-чата: {e}"

                    st.markdown(assistant_reply)

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="card">
    <b> Важно:</b><br>
    Это образовательный ML-проект,результат модели не является медицинским диагнозом 
    и не заменяет консультацию врача.
</div>
""", unsafe_allow_html=True)
