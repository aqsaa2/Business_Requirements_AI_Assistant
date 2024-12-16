import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import PyPDF2

# Set up Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'YOUR_HUGGING_FACE_API_TOKEN'

# Initialize Hugging Face model
model = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

# Define prompt templates
initial_prompt = PromptTemplate(
    input_variables=["context"],
    template="You are an intelligent agent helping with business requirements. Based on the context below, ask the most relevant questions to gather more details:\n\n{context}\n\n"
)

question_prompt = PromptTemplate(
    input_variables=["answers"],
    template="Based on these answers: {answers}, what follow-up questions should be asked to gather more details?"
)

user_story_prompt = PromptTemplate(
    input_variables=["answers"],
    template="Generate user stories and business scenarios based on the following details:\n\n{answers}\n\n"
)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function for dynamic conversation
def dynamic_conversation(context):
    # Generate initial questions
    llm_chain = LLMChain(prompt=initial_prompt, llm=model)
    questions = llm_chain.run(context)

    user_inputs = {}
    for question in questions.split("\n"):
        if question.strip():
            print(question)
            user_inputs[question] = input("> ")

    # Format the answers for the follow-up prompt
    answers = "\n".join([f"{q}: {a}" for q, a in user_inputs.items()])

    # Generate follow-up questions
    follow_up_chain = LLMChain(prompt=question_prompt, llm=model)
    follow_ups = follow_up_chain.run({"answers": answers})
    print("\nFollow-up Questions:")
    print(follow_ups)

    # Generate user stories and scenarios
    user_story_chain = LLMChain(prompt=user_story_prompt, llm=model)
    user_stories = user_story_chain.run({"answers": answers})
    print("\nGenerated User Stories and Business Scenarios:")
    print(user_stories)

# Main workflow
def main():
    file_path = input("Please provide the path to your PDF document: ")
    pdf_text = extract_text_from_pdf(file_path)

    if pdf_text.startswith("Error"):
        print(pdf_text)
        return

    print("PDF content successfully extracted.")

    # Start dynamic conversation with extracted context
    dynamic_conversation(pdf_text)

if __name__ == "__main__":
    main()
