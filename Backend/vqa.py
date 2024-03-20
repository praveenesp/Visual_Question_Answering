from transformers import AutoProcessor, AutoModelForCausalLM, BlipForQuestionAnswering, ViltForQuestionAnswering
import torch

git_processor_large = AutoProcessor.from_pretrained("microsoft/git-large-vqav2")
git_model_large = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vqav2")

def Vqa(image_path,question):
    def generate_answer_git(processor, model, image, question):
      pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # prepare question
      input_ids = processor(text=question, add_special_tokens=False).input_ids
      input_ids = [processor.tokenizer.cls_token_id] + input_ids
      input_ids = torch.tensor(input_ids).unsqueeze(0)

      generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
      generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)

      return generated_answer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    git_model_large.to(device)

    def generate_answers(image, question):
        answer_git_large = generate_answer_git(git_processor_large, git_model_large, image, question)
        return answer_git_large

    from PIL import Image

    # If you're loading the image from a local file, use:
    # image_path = "car.jpg"
    image = Image.open(image_path)


    # question="what is in the picture?"
    res=generate_answers(image,question)
    # print(res)
    return res

# ans=Vqa("car.jpg","what is in the picture?")
# print(ans)