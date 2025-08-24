import langextract as lx
import textwrap
import os

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""
Given RAP verses separated by empty lines, extract pairs of consecutive bars where the second bar rhymes with the first. For each pair, label the first bar as 'prompt' and the second as 'completion'. Skip any bar that does not have a rhyming completion. Process all verses in the input, and ignore the last bar of each verse if it does not have a rhyming pair.
""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="""\
            Let the suicide doors up
            I threw suicides on the tour bus""",
        extractions=[
            lx.data.Extraction(
                extraction_class="bar",
                extraction_text="Let the suicide doors up",
                attributes={"feeling": "accomplished"}
            ),
            lx.data.Extraction(
                extraction_class="completion",
                extraction_text="I threw suicides on the tour bus",
                attributes={"emotional_state": "flex"}
            )
        ]
    ),
    lx.data.ExampleData(
        text="""\
            Don't do no press but I get the most press kit
            Plus, yo, my bitch make your bitch look like Precious""",
        extractions=[
            lx.data.Extraction(
                extraction_class="bar",
                extraction_text="Don't do no press but I get the most press kit",
                attributes={"feeling": "confident"}
            ),
            lx.data.Extraction(
                extraction_class="completion",
                extraction_text="Plus, yo, my bitch make your bitch look like Precious",
                attributes={"emotional_state": "boastful"}
            )
        ]
    )
]

def main():
    # The input text to be processed (from file or fallback string)
    input_file_path = './my_kanye_data/kanye_verses_1.txt'
    if os.path.exists(input_file_path):
        with open(input_file_path, encoding="utf-8") as f:
            input_text = f.read()
    else:
        print(f"Warning: '{input_file_path}' not found. Using fallback input.")
        input_text = "Niggas is loiterin' just to feel important You gon' see lawyers and niggas in Jordans"

    print(input_text)
    try:
        # Run the extraction
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemma3:4b",
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False
        )
    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    try:
        # Save the results to a JSONL file
        lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
    except Exception as e:
        print(f"Saving results failed: {e}")
        return

    try:
        # Generate the visualization from the file
        html_content = lx.visualize("extraction_results.jsonl")
        with open("visualization.html", "w", encoding="utf-8") as f:
            # Check if html_content has 'data' attr or is a string
            if hasattr(html_content, 'data'):
                f.write(html_content.data)
            else:
                f.write(html_content)
        print("Visualization written to visualization.html")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
