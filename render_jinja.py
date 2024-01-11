from jinja2 import Template, Environment, FileSystemLoader
import os
from huggingface_hub import HfApi, file_exists

losses_ = ["ppo", "dpo", "kto", "sft", "slic", "sft+ppo", "sft+dpo", "sft+kto", "sft+slic", "sft+csft", "csft"]
models = ["pythia1-4b", "pythia2-8b", "pythia6-9b", "pythia12-0b", "llama7b", "llama13b", "llama30b"]

# Your list of losses (replace these with your actual data)
hf_link = "https://huggingface.co/ContextualAI/archangel"

losses = [{
    "pythia1-4b" : {"ppo": f"{hf_link}_ppo_pythia1-4b", "dpo": f"{hf_link}_dpo_pythia1-4b", "kto": f"{hf_link}_kto_pythia1-4b", "sft": f"{hf_link}_sft_pythia1-4b", "slic": f"{hf_link}_slic_pythia1-4b", "sft+ppo": f"{hf_link}_sft-ppo_pythia1-4b", "sft+dpo": f"{hf_link}_sft-dpo_pythia1-4b", "sft+kto": f"{hf_link}_sft-kto_pythia1-4b", "sft+csft": f"{hf_link}_sft_csft_pythia1-4b", "csft": f"{hf_link}_csft_pythia1-4b"},
    "pythia2-8b" : {"ppo": f"{hf_link}_ppo_pythia2-8b", "dpo": f"{hf_link}_dpo_pythia2-8b", "kto": f"{hf_link}_kto_pythia2-8b", "sft": f"{hf_link}_sft_pythia2-8b", "slic": f"{hf_link}_slic_pythia2-8b", "sft+ppo": f"{hf_link}_sft-ppo_pythia2-8b", "sft+dpo": f"{hf_link}_sft-dpo_pythia2-8b", "sft+kto": f"{hf_link}_sft-kto_pythia2-8b", "sft+csft": f"{hf_link}_sft_csft_pythia2-8b", "csft": f"{hf_link}_csft_pythia2-8b"},
    "pythia6-9b" : {"ppo": f"{hf_link}_ppo_pythia6-9b", "dpo": f"{hf_link}_dpo_pythia6-9b", "kto": f"{hf_link}_kto_pythia6-9b", "sft": f"{hf_link}_sft_pythia6-9b", "slic": f"{hf_link}_slic_pythia6-9b", "sft+ppo": f"{hf_link}_sft-ppo_pythia6-9b", "sft+dpo": f"{hf_link}_sft-dpo_pythia6-9b", "sft+kto": f"{hf_link}_sft-kto_pythia6-9b", "sft+csft": f"{hf_link}_sft_csft_pythia6-9b", "csft": f"{hf_link}_csft_pythia6-9b"},
    "pythia12-0b" : {"ppo": f"{hf_link}_ppo_pythia12-0b", "dpo": f"{hf_link}_dpo_pythia12-0b", "kto": f"{hf_link}_kto_pythia12-0b", "sft": f"{hf_link}_sft_pythia12-0b", "slic": f"{hf_link}_slic_pythia12-0b", "sft+ppo": f"{hf_link}_sft-ppo_pythia12-0b", "sft+dpo": f"{hf_link}_sft-dpo_pythia12-0b", "sft+kto": f"{hf_link}_sft-kto_pythia12-0b", "sft+csft": f"{hf_link}_sft_csft_pythia12-0b", "csft": f"{hf_link}_csft_pythia12-0b"},
    "llama7b" : {"ppo": f"{hf_link}_ppo_llama7b", "dpo": f"{hf_link}_dpo_llama7b", "kto": f"{hf_link}_kto_llama7b", "sft": f"{hf_link}_sft_llama7b", "slic": f"{hf_link}_slic_llama7b", "sft+ppo": f"{hf_link}_sft-ppo_llama7b", "sft+dpo": f"{hf_link}_sft-dpo_llama7b", "sft+kto": f"{hf_link}_sft-kto_llama7b", "sft+csft": f"{hf_link}_sft_csft_llama7b", "csft": f"{hf_link}_csft_llama7b"},
    "llama13b" : {"ppo": f"{hf_link}_ppo_llama13b", "dpo": f"{hf_link}_dpo_llama13b", "kto": f"{hf_link}_kto_llama13b", "sft": f"{hf_link}_sft_llama13b", "slic": f"{hf_link}_slic_llama13b", "sft+ppo": f"{hf_link}_sft-ppo_llama13b", "sft+dpo": f"{hf_link}_sft-dpo_llama13b", "sft+kto": f"{hf_link}_sft-kto_llama13b", "sft+csft": f"{hf_link}_sft_csft_llama13b", "csft": f"{hf_link}_csft_llama13b"},
    "llama30b" : {"ppo": f"{hf_link}_ppo_llama30b", "dpo": f"{hf_link}_dpo_llama30b", "kto": f"{hf_link}_kto_llama30b", "sft": f"{hf_link}_sft_llama30b", "slic": f"{hf_link}_slic_llama30b", "sft+ppo": f"{hf_link}_sft-ppo_llama30b", "sft+dpo": f"{hf_link}_sft-dpo_llama30b", "sft+kto": f"{hf_link}_sft-kto_llama30b", "sft+csft": f"{hf_link}_sft_csft_llama30b", "csft": f"{hf_link}_csft_llama30b"},
}]

# # Jinja template: the columns are each of the lossees, the rows are each of the models and the entries are the "weights" with the corresponding loss inside the "value" of each "model" in the dictionary linked as the url
template_str = """
| Model | PPO | DPO | KTO | SFT | SLIC | SFT+PPO | SFT+DPO | SFT+KTO | CSFT | SFT+CSFT |
| ------------- |:-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:| -------------:|-------------:|
{% for dict_item in losses -%}
{% for model, value in dict_item.items() %}| {{ model }} {% for loss, url in value.items() %}| [weights]({{ url }}) {% endfor %}|  
{% endfor %}
{% endfor %}
"""

# Create a Jinja template object to render the table of models
template = Template(template_str)
full_table = template.render(losses=losses)
print(full_table)

readme_template = "readme.jinja"
cond_readme_template = "readme_conditional.jinja"
env = Environment(
    loader=FileSystemLoader("/home/winnie/halos/")
)
template = env.get_template(readme_template)

def push_to_hub(api, readme_path, repo_name):
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=f"ContextualAI/{repo_name}",
        repo_type="model",
    )

for model in models:
    for loss in losses_:
        if 'csft' in loss: # if conditional model, use different template
            template = env.get_template(cond_readme_template)
        output = template.render(model=model, loss=loss.upper(), thumbnail="https://gist.github.com/assets/29318529/fe2d8391-dbd1-4b7e-9dc4-7cb97e55bc06")
        os.makedirs(f"/home/winnie/halos/model_cards/{model}_{loss}", exist_ok=True)
        with open(f"/home/winnie/halos/model_cards/{model}_{loss}/README.md", "w") as f:
            f.write(output)
print('finished rendering and writing out all the READMEs.')

count, total = 0, len(models) * len(losses_)
api = HfApi()
problem_repos = []
for model in models:
    for loss in losses_:
        readme_path = f"/home/winnie/halos/model_cards/{model}_{loss}/README.md"
        repo_name = f"archangel_{loss}_{model}"
        # replace + with -
        repo_name = repo_name.replace("+", "-")
        # if not file_exists(f'ContextualAI{repo_name}', "README.md"):
        try:
            push_to_hub(api, readme_path, repo_name)
            count += 1
            print(f"{count}/{total} Pushed {readme_path} to {repo_name}")
        except Exception as e:
            print(f"Failed to push {readme_path} to {repo_name} due to {e}")
            problem_repos.append(repo_name)
            continue
print(f"Finished pushing {count}/{total} model cards to the hub.")
print(f"Failed to push {len(problem_repos)} model cards to the hub: {problem_repos}")