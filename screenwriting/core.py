from pydantic import BaseModel, Field,  ConfigDict
from typing import List, Optional, Union
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import re
import json

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
    
    def add_paragraph(self, paragraph: Paragraph):
        self.paragraphs.append(paragraph)

    def add_dualdialogue(self, dual: DualDialogue):
        self.paragraphs.append(dual)


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

    def save_as_text(self, file_path: str, spacing=True, markdown=True):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.write_screenplay(spacing=spacing, markdown=markdown))

    def save_as_json(self, file_path: str):
        def serialize_text_element(te: TextElement):
            return {"text": te.text, "style": te.style}

        def serialize_paragraph(paragraph: Paragraph):
            return {
                "type": paragraph.type,
                "text_elements": [serialize_text_element(te) for te in paragraph.text_elements]
            }

        def serialize_dual_dialogue(dd: DualDialogue):
            return {
                "dual_dialogue": [serialize_paragraph(p) for p in dd.paragraphs]
            }

        def serialize_scene(scene: Scene):
            return {
                "paragraphs": [
                    serialize_dual_dialogue(p) if isinstance(p, DualDialogue)
                    else serialize_paragraph(p)
                    for p in scene.paragraphs
                ]
            }

        screenplay_data = {
            "scenes": [serialize_scene(scene) for scene in self.scenes]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(screenplay_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str) -> "Screenplay":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_json(data)

    @classmethod
    def _from_json(cls, data: dict) -> "Screenplay":
        def deserialize_text_element(te_data: dict) -> TextElement:
            return TextElement(text=te_data["text"], style=te_data.get("style"))

        def deserialize_paragraph(p_data: dict) -> Paragraph:
            return Paragraph(
                type=p_data["type"],
                text_elements=[deserialize_text_element(te) for te in p_data["text_elements"]]
            )

        def deserialize_dual_dialogue(dd_data: dict) -> DualDialogue:
            return DualDialogue(
                paragraphs=[deserialize_paragraph(p) for p in dd_data["dual_dialogue"]]
            )

        def deserialize_scene(scene_data: dict) -> Scene:
            paragraphs = []
            for p in scene_data["paragraphs"]:
                if "dual_dialogue" in p:
                    paragraphs.append(deserialize_dual_dialogue(p))
                else:
                    paragraphs.append(deserialize_paragraph(p))
            return Scene(paragraphs=paragraphs)

        screenplay = cls()
        for scene_data in data.get("scenes", []):
            screenplay.add_scene(deserialize_scene(scene_data))

        return screenplay


    @classmethod
    def from_fdx(cls, file_path: str) -> "Screenplay":
        def extract_text_elements(element: ET.Element) -> List[TextElement]:
            text_elements: List[TextElement] = []
            buffer = ''
            last_style = None

            for text_elem in element.findall('Text'):
                text = text_elem.text or ''
                style = text_elem.get('Style', '') or None
                if style != last_style and buffer:
                    text_elements.append(TextElement(text=buffer, style=last_style))
                    buffer = ''
                buffer += text
                last_style = style

            if buffer:
                text_elements.append(TextElement(text=buffer, style=last_style))

            return text_elements

        tree = ET.parse(file_path)
        root = tree.getroot()

        screenplay = cls()
        content = root.find('Content')
        current_scene = Scene(paragraphs=[])

        if content is not None:
            for element in content:
                if element.tag != 'Paragraph':
                    continue

                para_type = element.get('Type')
                dual_dialogue = element.find('DualDialogue')

                if para_type == "Scene Heading":
                    if current_scene.paragraphs:
                        screenplay.add_scene(current_scene)
                    current_scene = Scene(paragraphs=[])
                    current_scene.add_paragraph(Paragraph(
                        type="Scene Heading",
                        text_elements=extract_text_elements(element)
                    ))

                elif dual_dialogue is not None:
                    dual = []
                    for sub_element in dual_dialogue:
                        if sub_element.tag != 'Paragraph':
                            # Ignore non-paragraph artefacts (e.g., SpellCheckIgnoreLists)
                            continue
                        sub_type = sub_element.get('Type') or 'Action'
                        dual.append(Paragraph(
                            type=sub_type,
                            text_elements=extract_text_elements(sub_element)
                        ))
                    if dual:
                        current_scene.add_dualdialogue(DualDialogue(paragraphs=dual))

                else:
                    fallback_type = para_type or 'Action'
                    current_scene.add_paragraph(Paragraph(
                        type=fallback_type,
                        text_elements=extract_text_elements(element)
                    ))

            if current_scene.paragraphs:
                screenplay.add_scene(current_scene)

            # Extract metadata
            metadata_start = list(root).index(content) + 1
            metadata_elements = list(root)[metadata_start:]
            screenplay.set_metadata(metadata_elements)

        return screenplay
        
    @classmethod
    def from_plain(cls, file_path: str, markdown: bool = True) -> "Screenplay":
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return cls._from_lines(lines, markdown)

    @classmethod
    def _from_lines(cls, lines: List[str], markdown: bool = True, allow_fountain: bool = True, clean_spacing: bool = True, clean_paging: bool = True) -> "Screenplay":
        screenplay = cls()
        current_scene: List[Paragraph] = []
        previous_type = None

        def _clean_spacing(lines: List[str]) -> List[str]:
            cleaned = []
            prev_indent = None

            for line in lines:
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]

                if stripped == '':
                    cleaned.append('')
                    prev_indent = None
                    continue

                if cleaned and prev_indent is not None and (indent == prev_indent or indent == prev_indent + ' '):
                    cleaned[-1] = cleaned[-1].rstrip() + ' ' + stripped
                else:
                    cleaned.append(stripped if indent == '' else line)
                    prev_indent = indent

            return cleaned
        

        def _clean_paging(lines: List[str]) -> List[str]:
            cleaned = []
            pattern = re.compile(r'^\s*(?:p\.\s*|PAGE\s*|page\s*)?\d{1,3}\s*$')

            for line in lines:
                if pattern.match(line.strip()):
                    continue
                cleaned.append(line)
                
            return cleaned

        def parse_markdown(text: str, markdown: bool = True) -> List[TextElement]:
            if not markdown:
                return [TextElement(text=text)]

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

            elements: List[TextElement] = []
            text_remaining = text

            while text_remaining:
                for pattern, style in markdown_patterns:
                    match = re.search(pattern, text_remaining)
                    if match:
                        start, end = match.span()
                        if start > 0:
                            elements.append(TextElement(text=text_remaining[:start]))
                        elements.append(TextElement(text=match.group(1), style=style))
                        text_remaining = text_remaining[end:]
                        break
                else:
                    elements.append(TextElement(text=text_remaining))
                    break

            return elements


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
        
        def normalize_characters(paragraphs: List[Paragraph]):
            for i in range(len(paragraphs)):
                if paragraphs[i].type == "Character":
                    if i == len(paragraphs) - 1 or paragraphs[i + 1].type not in {"Dialogue", "Parenthetical"}:
                        paragraphs[i].type = "Action"

        if clean_spacing:
            lines = _clean_spacing(lines)

        if clean_paging:
            lines = _clean_paging(lines)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove leading scene numbers (e.g. '1A', '15.', '2', etc.)
            line_no_number = re.sub(r'^\s*\d+[A-Z]?[\.\s-]+', '', line)

            # Match potential scene headings
            if re.match(r'^(INT|EXT|INTERIOR|EXTERIOR|I\./E|I/E)[\s\./:-]', line_no_number) or (re.match(r'^\.(?!\.)', line) and allow_fountain):
                if current_scene:
                    normalize_characters(current_scene)
                    paragraphs = convert_dual_dialogues(current_scene)
                    screenplay.add_scene(Scene(paragraphs=paragraphs))
                current_scene = [Paragraph(
                    type="Scene Heading",
                    text_elements=parse_markdown(line_no_number.strip(), markdown)
                )]
                previous_type = "Scene Heading"
            elif line.startswith('@') and allow_fountain:
                current_scene.append(Paragraph(type="Character", text_elements=parse_markdown(line[1:], markdown)))
                previous_type = "Character"
            elif line.startswith('!') and allow_fountain:
                current_scene.append(Paragraph(type="Action", text_elements=parse_markdown(line[1:], markdown)))
                previous_type = "Action"
            elif line.startswith('>') and allow_fountain:
                current_scene.append(Paragraph(type="Transition", text_elements=parse_markdown(line[1:], markdown)))
                previous_type = "Transition"
            elif line.startswith('(') and line.endswith(')'):
                current_scene.append(Paragraph(type="Parenthetical", text_elements=parse_markdown(line, markdown)))
                previous_type = "Parenthetical"
            elif line.isupper() and line.endswith(':'):
                current_scene.append(Paragraph(type="Transition", text_elements=parse_markdown(line, markdown)))
                previous_type = "Transition"
            elif line.isupper():
                current_scene.append(Paragraph(type="Character", text_elements=parse_markdown(line, markdown)))
                previous_type = "Character"
            elif previous_type in {"Character", "Parenthetical"} and not (line.startswith('^') and allow_fountain):
                current_scene.append(Paragraph(type="Dialogue", text_elements=parse_markdown(line, markdown)))
                previous_type = "Dialogue"
            else:
                current_scene.append(Paragraph(type="Action", text_elements=parse_markdown(line, markdown)))
                previous_type = "Action"

        if current_scene:
            normalize_characters(current_scene)
            paragraphs = convert_dual_dialogues(current_scene)
            screenplay.add_scene(Scene(paragraphs=paragraphs))

        return screenplay

if __name__ == "__main__":
    test = 1
    if test == 1:
        input_fdx_file_path = "example.fdx"
        output_text_file_path_markdown = "example_script_markdown.txt"
        output_text_file_path_plain = "example_script_plain.txt"
        output_fdx_file_path = "recreated_example_script.fdx"
        read_without_markdown_path = "recreated_example_script_plain.fdx"
        json_file_path = "example_script.json"

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

        # Save the screenplay as a JSON file
        screenplay.save_as_json(json_file_path)
        

        print(f"Script has been saved as text with markdown to {output_text_file_path_markdown}")
        print(f"Script has been saved as text without markdown to {output_text_file_path_plain}")
        print(f"Script has been recreated as FDX to {output_fdx_file_path}")
    
    if test == 2:
    
        input_plain_file_path = "input_formatted.txt"
        output_file_path = "output.txt"
        screenplay = Screenplay.from_plain(input_plain_file_path, markdown=False)
        print(screenplay.scenes[1].write())
        screenplay.save_as_text(output_file_path, markdown=False)
