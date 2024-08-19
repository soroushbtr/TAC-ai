from flask import Flask, render_template, request
from rag_copy import answer_to_question


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        answer = f"Answer to your question: {answer_to_question(question)}"

    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run()