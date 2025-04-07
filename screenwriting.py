from pydantic import BaseModel, Field,  ConfigDict
from typing import List, Tuple, Optional, Union
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import re

from pprint import pprint

class TextElement(BaseModel):
    text: str
    style: Optional[str] = None

    def format(self, markdown: bool = True) -> str:
        if not markdown or not self.style:
            return self.text

        style_map = {
            'Bold+Italic+Underline+Strikeout': f"***~~__{self.text}__~~***",
            'Bold+Italic+Underline': f"***__{self.text}__***",
            'Bold+Italic+Strikeout': f"***~~{self.text}~~***",
            'Bold+Underline+Strikeout': f"**~~__{self.text}__~~**",
            'Italic+Underline+Strikeout': f"*~~__{self.text}__~~*",
            'Bold+Italic': f"***{self.text}***",
            'Bold+Underline': f"**__{self.text}__**",
            'Bold+Strikeout': f"**~~{self.text}~~**",
            'Italic+Underline': f"*__{self.text}__*",
            'Italic+Strikeout': f"*~~{self.text}~~*",
            'Underline+Strikeout': f"__~~{self.text}~~__",
            'Bold': f"**{self.text}**",
            'Italic': f"*{self.text}*",
            'Underline': f"__{self.text}__",
            'Strikeout': f"~~{self.text}~~",
        }

        return style_map.get(self.style, self.text)

class Paragraph(BaseModel):
    type: str
    text_elements: List[TextElement]

    def format(self, spacing: bool = True, markdown: bool = True) -> str:
        text = ''.join([e.format(markdown) for e in self.text_elements])

        if self.type == "Scene Heading":
            return text.upper()
        elif self.type == "Character":
            indent = " " * 21 if spacing else ""
            return f"{indent}{text.upper()}"
        elif self.type == "Parenthetical":
            indent = " " * 15 if spacing else ""
            return f"{indent}{text}"
        elif self.type == "Transition":
            indent = " " * (55 - len(text)) if spacing else ""
            return f"{indent}{text.upper()}"
        elif self.type == "Dialogue":
            indent = " " * 10 if spacing else ""
            return f"{indent}{text}"
        return text
    
    def fdx_paragraph_element(self) -> ET.Element:
        paragraph_element = ET.Element("Paragraph", {"Type": self.type})
        for elem in self.text_elements:
            text_element = ET.SubElement(paragraph_element, "Text")
            text_element.text = elem.text
            if elem.style:
                text_element.set("Style", elem.style)
        return paragraph_element

class DualDialogue(BaseModel):
    paragraphs: List[Paragraph]

    def format(self, spacing: bool = True, markdown: bool = True) -> str:
        formatted = []
        second_character = False

        for para in self.paragraphs:
            text = ''.join([e.format(markdown) for e in para.text_elements])
            para_type = para.type

            if para_type == "Character" and second_character and markdown:
                # Add ^ before second character
                text = f"^{text}"
                para = Paragraph(type=para_type, text_elements=[TextElement(text=text)])

            formatted.append(para.format(spacing, markdown))
            if para_type == "Character":
                second_character = True

        return '\n'.join(formatted)
    
    def fdx_paragraph_element(self) -> ET.Element:
        paragraph_element = ET.Element("Paragraph")
        dual_dialogue_element = ET.SubElement(paragraph_element, "DualDialogue")
        for para in self.paragraphs:
            dual_dialogue_element.append(para.fdx_paragraph_element())
        return paragraph_element



class Scene(BaseModel):
    paragraphs: List[Union[Paragraph, DualDialogue]]

    def write(self, spacing=True, markdown=False) -> str:
        scene_content = []
        previous_type = None

        for element in self.paragraphs:
            formatted_paragraph = element.format(spacing=spacing, markdown=markdown)

            current_type = (
                "DualDialogue" if isinstance(element, DualDialogue)
                else element.type
            )

            if spacing and previous_type and not (
                previous_type in {"Character", "Parenthetical", "Dialogue"}
                and current_type in {"Parenthetical", "Dialogue"}
            ):
                scene_content.append("")

            scene_content.append(formatted_paragraph)
            previous_type = current_type

        return '\n'.join(scene_content)


class Screenplay(BaseModel):
    scenes: List[Scene] = Field(default_factory=list)
    metadata: List[ET.Element] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_scene(self, scene: Scene):
        self.scenes.append(scene)

    def set_metadata(self, metadata: List[ET.Element]):
        self.metadata = metadata

    def extract_metadata_from_fdx(self, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()
        content = root.find('Content')

        if content is not None:
            metadata_start = list(root).index(content) + 1
            metadata_elements = [
                elem for elem in list(root)[metadata_start:]
                if elem.tag != 'LockedPages'
            ]
            self.set_metadata(metadata_elements)

    def write_screenplay(self, spacing=True, markdown=False) -> str:
        return '\n'.join([scene.write(spacing=spacing, markdown=markdown) for scene in self.scenes])

    def _save_as_fdx(self) -> str:
        root = ET.Element("FinalDraft", {"DocumentType": "Script", "Template": "No", "Version": "5"})
        content = ET.SubElement(root, "Content")

        for scene in self.scenes:
            for para in scene.paragraphs:
                content.append(para.fdx_paragraph_element())

        for elem in self.metadata:
            root.append(elem)

        xml_str = ET.tostring(root, encoding="unicode")
        parsed_str = minidom.parseString(xml_str)
        return parsed_str.toprettyxml(indent="  ")


    def save_as_fdx(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._save_as_fdx())

    def save_as_text(self, file_path: str, markdown=True):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.write_screenplay(spacing=True, markdown=markdown))

    @classmethod
    def from_fdx(cls, file_path: str) -> "Screenplay":
        def extract_text_elements(element: ET.Element) -> List[Tuple[str, Optional[str]]]:
            """Extract text and style elements from a Paragraph element."""
            text_elements = []
            buffer = ''
            last_style = None

            for text_elem in element.findall('Text'):
                text = text_elem.text or ''
                style = text_elem.get('Style', '')
                if style != last_style and buffer:
                    text_elements.append((buffer, last_style))
                    buffer = ''
                buffer += text
                last_style = style

            if buffer:
                text_elements.append((buffer, last_style))

            return text_elements
        
        tree = ET.parse(file_path)
        root = tree.getroot()

        screenplay = cls()
        content = root.find('Content')

        current_scene: List[Union[Paragraph, DualDialogue]] = []
        if content is not None:
            for element in content:
                if element.tag != 'Paragraph':
                    continue

                para_type = element.get('Type')
                dual_dialogue = element.find('DualDialogue')

                if para_type == "Scene Heading":
                    if current_scene:
                        cls._safe_add_scene(screenplay, current_scene)
                    current_scene = [Paragraph(
                        type="Scene Heading",
                        text_elements=to_text_elements(extract_text_elements(element))
                    )]

                elif dual_dialogue is not None:
                    dual = []
                    for sub_element in dual_dialogue:
                        sub_type = sub_element.get('Type')
                        dual.append(Paragraph(
                            type=sub_type,
                            text_elements=to_text_elements(extract_text_elements(sub_element))
                        ))
                    current_scene.append(DualDialogue(paragraphs=dual))

                else:
                    current_scene.append(Paragraph(
                        type=para_type,
                        text_elements=to_text_elements(extract_text_elements(element))
                    ))

            if current_scene:
                cls._safe_add_scene(screenplay, current_scene)

            # Extract metadata
            metadata_start = list(root).index(content) + 1
            metadata_elements = list(root)[metadata_start:]
            screenplay.set_metadata(metadata_elements)

        return screenplay


    @staticmethod
    def _safe_add_scene(screenplay: "Screenplay", current_scene):
        try:
            screenplay.add_scene(Scene(paragraphs=current_scene))
        except Exception as e:
            print("\n⛔️ Failed to create Scene with the following data:\n")
            pprint(current_scene)
            raise e
        
    @classmethod
    def from_plain(cls, file_path: str, markdown: bool = True) -> "Screenplay":
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return cls._from_lines(lines, markdown)

    @classmethod
    def _from_lines(cls, lines: List[str], markdown: bool = True) -> "Screenplay":
        screenplay = cls()
        current_scene: List[Paragraph] = []
        previous_type = None

        def parse_markdown(text: str, markdown: bool = True) -> List[Tuple[str, Optional[str]]]:
            if not markdown:
                return [(text, None)]

            markdown_patterns = [
                (r'\*\*\*~~__([^_~*]+?)__~~\*\*\*', 'Bold+Italic+Underline+Strikeout'),
                (r'\*\*\*__([^_]+?)__\*\*\*', 'Bold+Italic+Underline'),
                (r'\*\*\*~~([^~]+?)~~\*\*\*', 'Bold+Italic+Strikeout'),
                (r'\*\*~~__([^_~*]+?)__~~\*\*', 'Bold+Underline+Strikeout'),
                (r'\*~~__([^_~*]+?)__~~\*', 'Italic+Underline+Strikeout'),
                (r'\*\*\*([^*]+?)\*\*\*', 'Bold+Italic'),
                (r'\*\*__([^_]+?)__\*\*', 'Bold+Underline'),
                (r'\*\*~~([^~]+?)~~\*\*', 'Bold+Strikeout'),
                (r'\*__([^_]+?)__\*', 'Italic+Underline'),
                (r'\*~~([^~]+?)~~\*', 'Italic+Strikeout'),
                (r'__~~([^~]+?)~~__', 'Underline+Strikeout'),
                (r'\*\*([^*]+?)\*\*', 'Bold'),
                (r'\*([^*]+?)\*', 'Italic'),
                (r'__([^_]+?)__', 'Underline'),
                (r'~~([^~]+?)~~', 'Strikeout'),
            ]

            parts: List[Tuple[str, Optional[str]]] = []
            text_remaining = text

            while text_remaining:
                for pattern, style in markdown_patterns:
                    match = re.search(pattern, text_remaining)
                    if match:
                        start, end = match.span()
                        if start > 0:
                            parts.append((text_remaining[:start], None))
                        parts.append((match.group(1), style))
                        text_remaining = text_remaining[end:]
                        break
                else:
                    parts.append((text_remaining, None))
                    break

            return parts

        def convert_dual_dialogues(paragraphs: List[Paragraph]) -> List[Union[Paragraph, DualDialogue]]:
            updated = []
            i = 0
            while i < len(paragraphs):
                if (
                    isinstance(paragraphs[i], Paragraph) and
                    paragraphs[i].type == "Character" and
                    i + 3 < len(paragraphs) and
                    isinstance(paragraphs[i + 2], Paragraph) and
                    paragraphs[i + 2].type == "Character"
                ):
                    second_char_para = paragraphs[i + 2].text_elements
                    if second_char_para and second_char_para[0].text.startswith("^"):
                        # Remove ^ from second character name
                        first_element = second_char_para[0]
                        cleaned_name = first_element.text[1:]
                        cleaned_element = TextElement(text=cleaned_name, style=first_element.style)
                        new_second_char = Paragraph(type="Character", text_elements=[cleaned_element])

                        dual = DualDialogue(paragraphs=[
                            paragraphs[i], paragraphs[i + 1], new_second_char, paragraphs[i + 3]
                        ])
                        updated.append(dual)
                        i += 4
                        continue
                updated.append(paragraphs[i])
                i += 1
            return updated    

        def parsed(text: str) -> List[TextElement]:
            return to_text_elements(parse_markdown(text, markdown))
        
        def normalize_characters(paragraphs: List[Paragraph]):
            for i in range(len(paragraphs)):
                if paragraphs[i].type == "Character":
                    if i == len(paragraphs) - 1 or paragraphs[i + 1].type not in {"Dialogue", "Parenthetical"}:
                        paragraphs[i].type = "Action"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^(INT\.|EXT\.)', line) or (re.match(r'^\.(?!\.)', line) and markdown):
                if current_scene:
                    normalize_characters(current_scene)
                    paragraphs = convert_dual_dialogues(current_scene)
                    screenplay.add_scene(Scene(paragraphs=paragraphs))
                current_scene = [Paragraph(type="Scene Heading", text_elements=parsed(line))]
                previous_type = "Scene Heading"
            elif line.startswith('@') and markdown:
                current_scene.append(Paragraph(type="Character", text_elements=parsed(line[1:])))
                previous_type = "Character"
            elif line.startswith('!') and markdown:
                current_scene.append(Paragraph(type="Action", text_elements=parsed(line[1:])))
                previous_type = "Action"
            elif line.startswith('>') and markdown:
                current_scene.append(Paragraph(type="Transition", text_elements=parsed(line[1:])))
                previous_type = "Transition"
            elif line.startswith('(') and line.endswith(')'):
                current_scene.append(Paragraph(type="Parenthetical", text_elements=parsed(line)))
                previous_type = "Parenthetical"
            elif line.isupper() and line.endswith(':'):
                current_scene.append(Paragraph(type="Transition", text_elements=parsed(line)))
                previous_type = "Transition"
            elif line.isupper():
                current_scene.append(Paragraph(type="Character", text_elements=parsed(line)))
                previous_type = "Character"
            elif previous_type in {"Character", "Parenthetical"} and not line.startswith('^'):
                current_scene.append(Paragraph(type="Dialogue", text_elements=parsed(line)))
                previous_type = "Dialogue"
            else:
                current_scene.append(Paragraph(type="Action", text_elements=parsed(line)))
                previous_type = "Action"

        if current_scene:
            normalize_characters(current_scene)
            paragraphs = convert_dual_dialogues(current_scene)
            screenplay.add_scene(Scene(paragraphs=paragraphs))

        return screenplay

def to_text_elements(pairs: List[Tuple[str, Optional[str]]]) -> List[TextElement]:
    for item in pairs:
        if not isinstance(item, tuple) or not isinstance(item[0], str):
            raise ValueError(f"Invalid item in parse_markdown output: {item}")
    return [TextElement(text=t, style=s) for (t, s) in pairs]


if __name__ == "__main__":
    #input_fdx_file_path = "scènes intrigue B S.fdx"
    input_fdx_file_path = "example.fdx"
    output_text_file_path_markdown = "example_script_markdown.txt"
    output_text_file_path_plain = "example_script_plain.txt"
    output_fdx_file_path = "recreated_example_script.fdx"
    read_without_markdown_path = "recreated_example_script_plain.fdx"

    # Parse the FDX file
    screenplay = Screenplay.from_fdx(input_fdx_file_path)
    print(screenplay.scenes[1].write())
    # Save the screenplay as a text file with markdown
    screenplay.save_as_text(output_text_file_path_markdown, markdown=True)

    # Save the screenplay as a text file without markdown
    screenplay.save_as_text(output_text_file_path_plain, markdown=False)

    # Parse the text file with markdown
    screenplay_from_text = Screenplay.from_plain(output_text_file_path_markdown)

    # Extract metadata from the input FDX file and set it for the new screenplay
    screenplay_from_text.extract_metadata_from_fdx(input_fdx_file_path)

    # Save the screenplay parsed from text file as a new FDX file
    screenplay_from_text.save_as_fdx(output_fdx_file_path)

    print(f"Script has been saved as text with markdown to {output_text_file_path_markdown}")
    print(f"Script has been saved as text without markdown to {output_text_file_path_plain}")
    print(f"Script has been recreated as FDX to {output_fdx_file_path}")
