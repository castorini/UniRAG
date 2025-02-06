import argparse
import base64
from mimetypes import guess_type

from PIL import Image
from vertexai import generative_models



class Prompt:

    def __init__(self, prompt_file, k) -> None:
        self.k = k
        with open(prompt_file) as p:
            self.prompt_template = "".join(p.readlines()).strip()

    def prepare_llava_message(self, retrieval_results):
        examples = ""
        retrieval_results = retrieval_results or []
        num_examples = min(self.k, len(retrieval_results))
        if self.k > 0:
            index = 0
            for index, hit in enumerate(retrieval_results):
                if index == num_examples:
                    break
                examples += f"Caption [{index+1}]: {hit[0]}\n"
            examples += f"Caption [{index+1}]: ...\n"
            prompt = self.prompt_template.format(
                examples=examples, image_num=num_examples + 1, caption_num=num_examples
            )
        else:
            prompt = self.prompt_template
        return prompt

    def prepare_gpt_message(self, num_candidates):
        num_examples = min(self.k, num_candidates)
        if self.k > 0:
            prompt = self.prompt_template.format(num=num_examples)
        else:
            prompt = self.prompt_template
        return prompt
    
    def prepare_gemini_message(self, num_candidates):
        num_examples = min(self.k, num_candidates)
        if self.k > 0:
            prompt = self.prompt_template.format(num=num_examples)
        else:
            prompt = self.prompt_template.format(num=num_examples)
        return prompt

    def merge_images(self, retrieval_results, query_image_path, dist_images=5):
        if self.k == 0:
            image = Image.open(query_image_path)
            keep = image.copy()
            image.close()
            return keep

        images = []
        # Add in context examples first
        num_examples = min(self.k, len(retrieval_results))
        for index, hit in enumerate(retrieval_results):
            if index == num_examples:
                break
            image = Image.open(hit[1])
            keep = image.copy()
            image.close()
            images.append(keep)
        # Add query image as the last image
        image = Image.open(query_image_path)
        keep = image.copy()
        image.close()
        images.append(keep)

        # calc max width from imgs
        max_width = max(img.width for img in images)
        # calc total height of imgs + dist between them
        total_height = sum(img.height for img in images) + dist_images * (
            len(images) - 1
        )
        # create new img with calculated dimensions, black bg
        concat_img = Image.new("RGB", (max_width, total_height), (0, 0, 0))
        # init var to track current height pos
        current_height = 0
        for img in images:
            # paste img in concat_img at current height
            concat_img.paste(img, (0, current_height))
            # update current height for next img
            current_height += img.height + dist_images
        return concat_img

    def encode_image_as_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def get_fewshot_image_urls(self, retrieval_results):
        num_examples = min(self.k, len(retrieval_results))
        image_urls = []
        for index, hit in enumerate(retrieval_results):
            if index == num_examples:
                break
            assert hit[1]
            image_urls.append(self.encode_image_as_url(hit[1]))
        return image_urls
    
    def get_fewshot_image_data(self, retrieval_results):
        num_examples = min(self.k, len(retrieval_results))
        image_datas = []
        for index, hit in enumerate(retrieval_results):
            if index == num_examples:
                break
            assert hit[1]
            temp = generative_models.Image.load_from_file(hit[1])
            image_datas.append(generative_models.Part.from_image(temp))
        return image_datas

    def get_fewshot_captions(self, retrieval_results):
        num_examples = min(self.k, len(retrieval_results))
        captions = []
        for index, hit in enumerate(retrieval_results):
            if index == num_examples:
                break
            assert hit[0]
            captions.append(hit[0])
        return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default=False, help="Prompt file")
    parser.add_argument(
        "--k", default=0, help="Number of retrieved examples included in the prompt"
    )

    args = parser.parse_args()

    retrieval_results = {"hits": []}

    p = Prompt(args.prompt_file, args.k)
    print(p.prepare_message(retrieval_results))


if __name__ == "__main__":
    main()
