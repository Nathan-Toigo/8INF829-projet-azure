"""
Interface Streamlit pour l'agent clinique.
Lancer: streamlit run app.py
"""
import asyncio
import json
import streamlit as st

from agent import ClinicalAgent


# ---------- Configuration de page ----------
st.set_page_config(
    page_title="Agent Clinique - Preparation de consultation",
    page_icon=":hospital:",
    layout="wide",
)

TOOL_LABELS = {
    "rag__search_clinical_documents": ("Recherche dans le dossier", "#1C7293"),
    "timeline__get_patient_timeline": ("Chronologie patient", "#2C5F2D"),
    "summary__get_patient_summary": ("Resume patient", "#B85042"),
    "risk__get_clinical_risks": ("Analyse de risques", "#6D2E46"),
}

SUGGESTED_QUESTIONS = [
    "Prepare-moi un point clinique synthetique pour la consultation de ce patient.",
    "Quelle est l'evolution de la CRP du patient sur les deux dernieres annees?",
    "Quels examens sont en retard ou recommandes mais non realises?",
    "Quel a ete le resultat de la biopsie ganglionnaire de 2021?",
    "Quand le patient a-t-il eu son scanner thoracique et qu'est-ce qu'il a montre?",
]


# ---------- Boucle asyncio persistante ----------
@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


@st.cache_resource
def get_agent():
    """Demarre l'agent une seule fois pour toute la session Streamlit."""
    loop = get_event_loop()
    agent = ClinicalAgent()
    with st.spinner("Demarrage des outils MCP (peut prendre 30 secondes)..."):
        loop.run_until_complete(agent.start())
    return agent


def ask_sync(agent: ClinicalAgent, question: str):
    loop = get_event_loop()
    return loop.run_until_complete(agent.ask(question))


# ---------- UI ----------
st.title("Agent Clinique")
st.caption("Preparation de consultation a partir du dossier patient (RAG + outils MCP + agent orchestrateur)")

# Sidebar
with st.sidebar:
    st.header("A propos")
    st.markdown(
        """
**Architecture**

L'agent dispose de 4 outils MCP qu'il combine selon la question :

- **Recherche dossier** : RAG hybride (dense + BM25 + reranker) sur les documents cliniques
- **Chronologie** : extraction des evenements dates
- **Resume patient** : synthese pre-calculee du dossier
- **Analyse de risques** : red flags pre-calcules par priorite

**Modele de generation** : Azure OpenAI gpt-4.1-mini
"""
    )
    st.divider()
    st.header("Questions suggerees")
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, key=f"sugg_{q[:30]}", use_container_width=True):
            st.session_state["pending_question"] = q

# Initialisation de l'agent
agent = get_agent()
st.success(f"Agent pret. {len(agent.openai_tools)} outils disponibles : "
           + ", ".join(agent.clients.keys()))

# Zone de saisie
default_q = st.session_state.pop("pending_question", "")
question = st.text_area(
    "Pose ta question clinique :",
    value=default_q,
    height=80,
    placeholder="Ex : Quels sont les elements importants a aborder avec le medecin?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Envoyer", type="primary", use_container_width=True)

# Traitement
if submit and question.strip():
    with st.spinner("L'agent reflechit..."):
        answer, trace = ask_sync(agent, question.strip())

    # Affichage de la reponse
    st.divider()
    st.subheader("Reponse")
    st.markdown(answer)

    # Affichage du raisonnement
    st.divider()
    st.subheader("Raisonnement de l'agent")
    tool_calls = [s for s in trace if s.kind == "tool_call"]
    if not tool_calls:
        st.info("L'agent a repondu sans appeler d'outil.")
    else:
        st.caption(f"L'agent a effectue {len(tool_calls)} appel(s) d'outil :")
        # On groupe call + result
        steps = []
        i = 0
        while i < len(trace):
            if trace[i].kind == "tool_call":
                call = trace[i]
                result = trace[i + 1] if i + 1 < len(trace) and trace[i + 1].kind == "tool_result" else None
                steps.append((call, result))
                i += 2 if result else 1
            else:
                i += 1

        for idx, (call, result) in enumerate(steps, start=1):
            label, color = TOOL_LABELS.get(call.tool_name, (call.tool_name, "#666666"))
            with st.expander(f"Etape {idx} : {label}", expanded=False):
                st.markdown(f"**Outil appele** : `{call.tool_name}`")
                st.markdown("**Arguments** :")
                st.json(call.arguments)
                if result:
                    st.markdown(f"**Resultat** ({len(result.content)} caracteres) :")
                    # Si le resultat ressemble a du JSON on le formate
                    try:
                        parsed = json.loads(result.content)
                        st.json(parsed)
                    except (json.JSONDecodeError, ValueError):
                        st.text(result.content[:3000] + ("..." if len(result.content) > 3000 else ""))

elif submit:
    st.warning("Tape une question avant d'envoyer.")