# OmniVoice Voice Design Attributes & Non-Verbal Tags
# A documentação oficial do OmniVoice não lista "emoções" puras (como "feliz", "triste"). 
# Em vez disso, ele controla a expressividade de duas formas:
# 1. Símbolos não-verbais (inseridos no meio ou início do texto)
# 2. Atributos da voz (passados no parâmetro `instruct`)

# ==========================================
# 1. SÍMBOLOS NÃO-VERBAIS (Tags de Emoção/Reação)
# Devem ser concatenados no texto. Ex: f"{random.choice(NON_VERBAL_TAGS)} {phrase}"
# ==========================================
NON_VERBAL_TAGS = [
    "",  # Opção neutra (sem tag)
    "[laughter]",             # Risada
    "[sigh]",                 # Suspiro
    "[surprise-ah]",          # Surpresa (Ah!)
    "[surprise-oh]",          # Surpresa (Oh!)
    "[surprise-wa]",          # Surpresa (Wa!)
    "[surprise-yo]",          # Surpresa (Yo!)
    "[dissatisfaction-hnn]",  # Insatisfação (Hnn...)
    "[confirmation-en]",      # Confirmação (En)
    "[question-en]",          # Dúvida/Pergunta (En?)
    "[question-ah]",          # Dúvida/Pergunta (Ah?)
    "[question-oh]",          # Dúvida/Pergunta (Oh?)
    "[question-ei]",          # Dúvida/Pergunta (Ei?)
    "[question-yi]"           # Dúvida/Pergunta (Yi?)
]

# ==========================================
# 2. ATRIBUTOS DA VOZ (Voice Design)
# Passados no parâmetro `instruct` separados por vírgula.
# Ex: instruct = f"{random.choice(GENDER)}, {random.choice(AGE)}, {random.choice(PITCH)}"
# ==========================================
GENDER = ["male", "female"]

AGE = ["", "child", "teenager", "young adult", "middle-aged", "elderly"]

PITCH = ["", "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"]

# OmniVoice atualmente só suporta formalmente "whisper" (sussurro) como estilo.
STYLE = ["", "whisper"] 

ACCENTS = [
    "", # Sem sotaque forçado
    "american accent", 
    "british accent", 
    "australian accent", 
    "canadian accent", 
    "indian accent", 
    "chinese accent", 
    "korean accent", 
    "japanese accent", 
    "portuguese accent", 
    "russian accent"
]
