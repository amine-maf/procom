def generate_topics_for_cluster(messages_cluster):
    prompt = f"Donne les principaux sujets ou thèmes pour les messages suivants :\n\n" + "\n".join(messages_cluster) + "\n\nQuels sont les topics principaux de ces messages ?"
    # Ancienne syntaxe pour la version 0.29
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Vérifiez que ce modèle est disponible avec votre clé
        messages=[
            {"role": "system", "content": "Vous êtes un assistant qui aide à générer des topics à partir de textes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    topics = response['choices'][0]['message']['content'].strip()
    return topics
