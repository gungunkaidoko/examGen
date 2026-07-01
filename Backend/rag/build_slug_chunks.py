"""
build_slug_chunks.py
====================
One-time script that:
  1. Parses ccc_book.txt into subtopic-scoped leaf chunks
  2. Assigns canonical topic_slug, chapter_id, bloom_levels, difficulty
  3. Writes enriched_chunks.jsonl  (~183 chunks)

Run once:
    python3 -m rag.build_slug_chunks

The output file is then fed to ingest_slugs.py which embeds + upserts to Pinecone.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BOOK_TXT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "Knowledge_base", "book", "ccc_book.txt")
OUT_JSONL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "Knowledge_base", "book", "enriched_chunks.jsonl")

# ── Exact prose-start line numbers (detected from 'Objective of the Chapter' pattern) ──
# Each chapter's prose body starts at the line BEFORE 'Objective of the Chapter' - 2
CHAPTER_PROSE_STARTS = {
    "ch1": (337,  1721, "introduction_to_computer"),
    "ch2": (1721, 2593, "introduction_to_operating_system"),
    "ch3": (2593, 3997, "word_processing"),
    "ch4": (3997, 5207, "spreadsheet"),
    "ch5": (5207, 6229, "presentation"),
    "ch6": (6229, 7475, "internet_www"),
    "ch7": (7475, 8449, "email_social_egov"),
    "ch8": (8449, 9261, "digital_financial_tools"),
    "ch9": (9261, None, "futureskills_cybersecurity"),
}

# ── Canonical topic slugs per chapter ─────────────────────────────────────
# Each entry: (heading_pattern_regex, slug, bloom_levels, difficulty_levels)
SLUG_MAP: dict[str, list[tuple]] = {
    "ch1": [
        (r"What is Computer|Introduction\b|Objective",  "computer_basics",        ["remember"],                ["easy"]),
        (r"Evolution|Generation",                        "computer_generations",   ["remember", "understand"],  ["easy", "medium"]),
        (r"IT Gadget|Tablet|Smartphone",                 "it_gadgets",             ["remember"],                ["easy"]),
        (r"Input Device|Keyboard|Mouse|Scanner",         "input_devices",          ["remember", "understand"],  ["easy", "medium"]),
        (r"Output Device|Monitor|Printer|Plotter",       "output_devices",         ["remember", "understand"],  ["easy", "medium"]),
        (r"CPU|Central Processing|ALU|Control Unit",     "cpu_architecture",       ["understand", "analyze"],   ["medium", "hard"]),
        (r"Memory|RAM|ROM|Cache|Storage",                "memory_storage",         ["remember", "understand"],  ["easy", "medium"]),
        (r"Software|Application|System|Open Source",     "software_types",         ["understand", "apply"],     ["medium"]),
        (r"Mobile App",                                  "mobile_apps",            ["remember"],                ["easy"]),
        (r"Number System|Binary|Decimal|Hexadecimal",    "number_systems",         ["understand", "apply"],     ["medium", "hard"]),
        (r"Network|LAN|WAN|MAN",                         "basic_networking",       ["remember"],                ["easy"]),
    ],
    "ch2": [
        (r"Operating System|OS\b|Basics of OS|Objective","os_basics",              ["remember", "understand"],  ["easy", "medium"]),
        (r"Windows|Linux|Android|Desktop OS|Mobile OS",  "os_types",               ["remember"],                ["easy"]),
        (r"Task Bar|Icons|Shortcuts|Desktop",            "desktop_ui",             ["remember", "apply"],       ["easy"]),
        (r"Mouse|Changing.*Properties",                  "mouse_settings",         ["apply"],                   ["easy"]),
        (r"Date and Time|Display Properties",            "system_settings",        ["apply"],                   ["easy"]),
        (r"Add.*Remove|Program.*Feature",                "software_management",    ["apply"],                   ["easy", "medium"]),
        (r"Printer|Adding.*Printer",                     "printer_management",     ["apply"],                   ["easy"]),
        (r"File.*Folder|File Extension|Folder.*Manage",  "file_folder_management", ["remember", "apply"],       ["easy", "medium"]),
    ],
    "ch3": [
        (r"Objective|Introduction|What is Writer",        "writer_overview",        ["remember"],                ["easy"]),
        (r"Document Creation|New Document|Opening",       "document_creation",      ["apply"],                   ["easy"]),
        (r"Save|Save As|PDF|Closing",                     "save_document",          ["apply"],                   ["easy"]),
        (r"Page Setup|Print Preview|Printing",            "page_setup_print",       ["apply"],                   ["easy"]),
        (r"Cut.*Copy.*Paste|Clipboard",                   "cut_copy_paste",         ["remember", "apply"],       ["easy"]),
        (r"Font|Color|Style|Size.*Selection",             "font_formatting",        ["apply"],                   ["easy"]),
        (r"Alignment|Justify|Center|Left.*Right",         "text_alignment",         ["apply"],                   ["easy"]),
        (r"Undo|Redo",                                    "undo_redo",              ["remember"],                ["easy"]),
        (r"AutoCorrect|Spelling|Grammar",                 "autocorrect_spelling",   ["apply"],                   ["easy"]),
        (r"Find.*Replace",                                "find_replace",           ["apply"],                   ["easy", "medium"]),
        (r"Paragraph.*Indent|Bullets|Numbering",          "paragraph_formatting",   ["apply"],                   ["easy", "medium"]),
        (r"Change Case|Header.*Footer",                   "header_footer",          ["apply"],                   ["easy"]),
        (r"Table|Insert.*Table|Table.*Row",               "tables_in_writer",       ["apply", "analyze"],        ["medium"]),
        (r"Mail Merge",                                   "mail_merge",             ["understand", "apply"],     ["medium", "hard"]),
        (r"Track Change|Comments|Review",                 "track_changes",          ["apply"],                   ["medium"]),
        (r"Macro",                                        "macros_writer",          ["understand", "apply"],     ["hard"]),
        (r"Shortcut|Keyboard.*Key|Ctrl\+",                "shortcut_keys_writer",   ["remember"],                ["easy"]),
        (r"Editing Text|Text Selection",                  "editing_text",           ["apply"],                   ["easy"]),
    ],
    "ch4": [
        (r"Objective|Introduction|What is Calc",          "calc_overview",          ["remember"],                ["easy"]),
        (r"Spreadsheet|Cell.*Address|Row.*Column",        "spreadsheet_basics",     ["remember", "understand"],  ["easy"]),
        (r"Entering Data|Data.*Cell|Input.*Cell",         "entering_data",          ["apply"],                   ["easy"]),
        (r"Formula|Function|SUM|AVERAGE|COUNT",           "formulas_functions",     ["apply", "analyze"],        ["medium", "hard"]),
        (r"Chart|Graph|Bar|Pie",                          "charts_graphs",          ["apply"],                   ["medium"]),
        (r"Sort|Filter|Data.*Sort",                       "sort_filter",            ["apply", "analyze"],        ["medium"]),
        (r"Format.*Cell|Number Format|Currency",          "cell_formatting",        ["apply"],                   ["easy", "medium"]),
        (r"Page Setup|Print.*Sheet|Printing",             "printing_spreadsheet",   ["apply"],                   ["easy"]),
        (r"Freeze|Split|Pane",                            "freeze_panes",           ["apply"],                   ["medium"]),
        (r"Shortcut|Ctrl\+",                              "shortcut_keys_calc",     ["remember"],                ["easy"]),
        (r"IF|VLOOKUP|HLOOKUP|Logical",                   "advanced_functions",     ["analyze"],                 ["hard"]),
        (r"Workbook|Sheet.*Tab|Multiple.*Sheet",          "workbook_sheets",        ["understand", "apply"],     ["easy", "medium"]),
    ],
    "ch5": [
        (r"Objective|Introduction|What is Impress",       "impress_overview",       ["remember"],                ["easy"]),
        (r"Presentation|New.*Presentation|Starting",      "presentation_basics",    ["remember", "understand"],  ["easy"]),
        (r"Template|Blank Presentation",                  "presentation_template",  ["apply"],                   ["easy"]),
        (r"Insert.*Text|Edit.*Text|Text.*Slide",          "slide_text_editing",     ["apply"],                   ["easy"]),
        (r"Insert.*Slide|Delete.*Slide|Slide.*Order",     "slide_management",       ["apply"],                   ["easy"]),
        (r"Animation|Transition|Slide.*Show",             "animation_transition",   ["apply"],                   ["medium"]),
        (r"Insert.*Image|Insert.*Picture|ClipArt",        "inserting_media",        ["apply"],                   ["easy"]),
        (r"Table.*Slide|Insert.*Table",                   "tables_in_impress",      ["apply"],                   ["medium"]),
        (r"Master.*Slide|Slide.*Layout",                  "slide_master_layout",    ["understand", "apply"],     ["medium"]),
        (r"Saving.*Presentation|Export.*PDF",             "saving_presentation",    ["apply"],                   ["easy"]),
    ],
    "ch6": [
        (r"Objective|Introduction|What is Internet",      "internet_overview",      ["remember"],                ["easy"]),
        (r"History.*Internet|Evolution.*Internet",        "internet_history",       ["remember"],                ["easy"]),
        (r"WWW|World Wide Web|HTTP|URL|Browser",          "www_basics",             ["remember", "understand"],  ["easy"]),
        (r"LAN|WAN|MAN|Network.*Type|PAN\b",              "network_types",          ["remember"],                ["easy"]),
        (r"IP Address|Domain|DNS",                        "ip_domain",              ["understand"],              ["medium"]),
        (r"Search Engine|Google|Browsing|Web Search",     "search_engines",         ["apply"],                   ["easy"]),
        (r"Download|Upload|FTP",                          "download_upload",        ["apply"],                   ["easy"]),
        (r"Wi-Fi|Wireless|Bluetooth|Hotspot",             "wireless_networking",    ["understand"],              ["easy", "medium"]),
        (r"Firewall|Proxy|VPN",                           "firewall_proxy_vpn",     ["understand", "analyze"],   ["medium", "hard"]),
        (r"Cloud|Cloud Computing|SaaS|IaaS",              "cloud_computing_basics", ["understand"],              ["medium"]),
        (r"IoT|Internet of Things",                       "iot_basics",             ["understand"],              ["medium"]),
        (r"E-commerce|Online Shopping|Digital Market",    "ecommerce",              ["understand", "apply"],     ["medium"]),
        (r"Chatting|Video Conferencing|Skype",            "online_communication",   ["apply"],                   ["easy"]),
    ],
    "ch7": [
        (r"Objective|Introduction|What is Email",         "email_overview",         ["remember"],                ["easy"]),
        (r"E-?[Mm]ail|Opening.*Account|Email.*Account",   "email_basics",           ["remember", "apply"],       ["easy"]),
        (r"Inbox|Outbox|Sent|Draft|Mailbox",              "email_folders",          ["remember"],                ["easy"]),
        (r"Creating.*E-?mail|Sending.*E-?mail|Compose",   "composing_email",        ["apply"],                   ["easy"]),
        (r"Reply|Forward|Attachment",                     "email_reply_forward",    ["apply"],                   ["easy"]),
        (r"Subject|CC|BCC|To.*Field",                     "email_fields",           ["remember"],                ["easy"]),
        (r"Social Network|Facebook|Twitter|LinkedIn",     "social_networking",      ["remember", "understand"],  ["easy"]),
        (r"Blog|Wiki|RSS",                                "blogs_wikis",            ["understand"],              ["easy", "medium"]),
        (r"e-Governance|Digital India|Government.*Service","egovernance",           ["understand", "apply"],     ["medium"]),
        (r"IRCTC|Railway|Online.*Booking",                "irctc_railway",          ["apply"],                   ["medium"]),
        (r"Cyber Crime|Online Safety|Phishing|Spam",      "cyber_safety_email",     ["understand", "analyze"],   ["medium", "hard"]),
    ],
    "ch8": [
        (r"Objective|Introduction|Financial.*Tool",       "digital_finance_overview",["remember"],               ["easy"]),
        (r"OTP|One Time Password",                        "otp_authentication",     ["remember", "understand"],  ["easy"]),
        (r"QR Code|Quick Response",                       "qr_code_payment",        ["remember", "understand"],  ["easy"]),
        (r"UPI|Unified Payment|BHIM",                     "upi_payment",            ["remember", "understand"],  ["easy", "medium"]),
        (r"AEPS|Aadhaar.*Payment|Aadhaar Enabled",        "aeps_biometric",         ["understand"],              ["medium"]),
        (r"USSD|\*99#|Unstructured Supplementary",        "ussd_99_banking",        ["remember", "understand"],  ["medium"]),
        (r"Credit.*Card|Debit.*Card|Card.*Payment",       "debit_credit_card",      ["remember", "understand"],  ["easy", "medium"]),
        (r"eWallet|Digital Wallet|Paytm|PhonePe",         "ewallet",                ["understand", "apply"],     ["easy", "medium"]),
        (r"PoS|Point.*Sale|POS Terminal",                 "pos_terminal",           ["remember"],                ["easy"]),
        (r"NEFT|National Electronic Fund",                "neft_transfer",          ["remember", "understand"],  ["medium"]),
        (r"RTGS|Real Time Gross",                         "rtgs_transfer",          ["understand"],              ["medium", "hard"]),
        (r"IMPS|Immediate Payment",                       "imps_transfer",          ["understand"],              ["medium"]),
        (r"Online.*Bill|Bill.*Payment",                   "online_bill_payment",    ["apply"],                   ["easy"]),
        (r"Internet Banking|Net Banking",                 "internet_banking",       ["understand", "apply"],     ["medium"]),
    ],
    "ch9": [
        (r"Objective|Introduction|FutureSkill",           "futureskills_overview",  ["remember"],                ["easy"]),
        (r"IoT|Internet of Things",                       "iot_futureskills",       ["understand"],              ["medium"]),
        (r"Big Data|Analytics",                           "big_data",               ["understand"],              ["medium"]),
        (r"Cloud Computing|IaaS|PaaS|SaaS",               "cloud_computing",        ["understand", "analyze"],   ["medium", "hard"]),
        (r"Virtual Reality|VR|AR|Augmented",              "virtual_reality",        ["understand"],              ["medium"]),
        (r"Artificial Intelligence|AI|Machine Learning",  "ai_ml",                  ["understand", "analyze"],   ["medium", "hard"]),
        (r"Blockchain|Distributed Ledger",                "blockchain",             ["understand"],              ["medium", "hard"]),
        (r"3D Print|Additive Manufacturing",              "3d_printing",            ["understand"],              ["medium"]),
        (r"Robotics|Automation|RPA",                      "robotics_rpa",           ["understand"],              ["medium"]),
        (r"Social.*Mobile|Social Media.*Platform",        "social_mobile",          ["understand"],              ["easy", "medium"]),
        (r"Cyber Security|Cyber.*Attack|Need.*Security",  "cybersecurity_basics",   ["understand", "analyze"],   ["medium"]),
        (r"Securing.*PC|Antivirus|Firewall",              "securing_pc",            ["apply"],                   ["medium"]),
        (r"Securing.*Smart[Pp]hone|Mobile.*Security",     "securing_smartphone",    ["apply"],                   ["medium"]),
        (r"Privacy|Data Protection|Password",             "privacy_data_protection",["apply", "analyze"],        ["medium", "hard"]),
    ],
}

CROSS_REF = {
    "shortcut_keys_writer": {"home": "ch3", "cross_ref": ["ch4", "ch5"]},
    "save_document":        {"home": "ch3", "cross_ref": ["ch4", "ch5"]},
    "tables_in_writer":     {"home": "ch3", "cross_ref": ["ch5"]},
    "cloud_computing_basics":{"home":"ch6", "cross_ref": ["ch9"]},
    "irctc_railway":        {"home": "ch7", "cross_ref": []},
    "cyber_safety_email":   {"home": "ch7", "cross_ref": ["ch9"]},
    "iot_basics":           {"home": "ch6", "cross_ref": ["ch9"]},
}


def _slug_for_heading(chapter_id: str, text: str) -> tuple | None:
    for pattern, slug, bloom, diff in SLUG_MAP.get(chapter_id, []):
        if re.search(pattern, text, re.IGNORECASE):
            return slug, bloom, diff
    return None


def _classify_chunk_type(text: str) -> str:
    if re.search(r"Ctrl\+|Alt\+|shortcut|keyboard", text, re.IGNORECASE):
        return "shortcut"
    if len(text) < 300 and re.search(r"summary|objective|overview|intro", text, re.IGNORECASE):
        return "summary"
    return "leaf"


def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    keywords = set()
    keywords.update(re.findall(r"\b[A-Z]{2,6}\b", text))
    keywords.update(re.findall(r"\b[A-Z][a-z]{3,15}\b", text))
    stop = {"This", "The", "That", "With", "When", "What", "Where", "Which",
            "Some", "Each", "They", "Your", "From", "Into", "Over", "Also",
            "Used", "Have", "Been", "Will", "Can", "Its", "For", "Are"}
    keywords -= stop
    keywords.update(re.findall(r"Ctrl\+\w+", text))
    return sorted(keywords)[:max_kw]


def _is_heading(line: str) -> bool:
    """Detect a new section heading line."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if s.startswith("-") or s.startswith("|") or s.startswith("●"):
        return False
    if re.match(r"^\d+\.\s+\d+", s):   # section number like "3.4.1 ..."
        return False
    if not re.search(r"[A-Za-z]{3,}", s):
        return False
    # Must be title-case or all-caps start, not a sentence
    if s[0].islower():
        return False
    return True


def parse_book_to_chunks(book_txt_path: str) -> list[dict]:
    with open(book_txt_path, encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    chunk_counter = 0

    for chapter_id, (start, end, chapter_name_snake) in CHAPTER_PROSE_STARTS.items():
        end_line = end if end is not None else len(lines)
        chapter_lines = lines[start:end_line]

        # Split into sections by heading detection
        sections: list[tuple[str, list[str]]] = []
        current_heading = chapter_lines[0].strip() if chapter_lines else "Introduction"
        current_lines: list[str] = []

        for line in chapter_lines[1:]:
            if _is_heading(line) and len(current_lines) > 3:
                text_so_far = "".join(current_lines).strip()
                if len(text_so_far) >= 60:
                    sections.append((current_heading, current_lines))
                current_heading = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            text_so_far = "".join(current_lines).strip()
            if len(text_so_far) >= 60:
                sections.append((current_heading, current_lines))

        # Build one chunk per section
        for section_heading, section_lines in sections:
            text = "".join(section_lines).strip()
            if len(text) < 80:
                continue

            # Assign slug: try heading first, then first 300 chars of text
            slug_result = _slug_for_heading(chapter_id, section_heading)
            if slug_result is None:
                slug_result = _slug_for_heading(chapter_id, text[:300])
            if slug_result is None:
                slug = re.sub(r"[^a-z0-9]+", "_", section_heading.lower())[:40].strip("_")
                bloom = ["remember"]
                difficulty = ["easy"]
            else:
                slug, bloom, difficulty = slug_result

            chunk_type = _classify_chunk_type(text)
            chunk_id   = f"{chapter_id}_{slug}_{chunk_counter:04d}"
            cross      = CROSS_REF.get(slug, {})
            # concept_id = stable book-agnostic identifier: chapter_id:topic_slug
            concept_id = f"{chapter_id}:{slug}"

            chunks.append({
                "chunk_id":         chunk_id,
                "chapter_id":       chapter_id,
                "chapter_name":     chapter_name_snake,
                "section_heading":  section_heading,
                "topic_slug":       slug,
                "concept_id":       concept_id,
                "book_id":          "ccc_arihant_v1",
                "chunk_type":       chunk_type,
                "bloom_levels":     bloom,
                "difficulty":       difficulty,
                "keywords":         _extract_keywords(text),
                "used_in_sets":     [],
                "q_count":          0,
                "home_chapter":     cross.get("home", chapter_id),
                "cross_ref":        cross.get("cross_ref", []),
                "text":             text,
            })
            chunk_counter += 1

    return chunks




def main():
    import json as _json
    print("Parsing book:", BOOK_TXT)
    chunks = parse_book_to_chunks(BOOK_TXT)
    print("Generated", len(chunks), "chunks")

    by_chapter: dict = {}
    by_slug: dict = {}
    for c in chunks:
        by_chapter[c["chapter_id"]] = by_chapter.get(c["chapter_id"], 0) + 1
        by_slug[c["topic_slug"]]    = by_slug.get(c["topic_slug"], 0) + 1

    print("\nChunks per chapter:")
    for ch, cnt in sorted(by_chapter.items()):
        print("  " + ch + ": " + str(cnt))
    print("Unique slugs:", len(by_slug))

    print("\nSample chunks:")
    for c in chunks[:4]:
        preview = c["text"][:80].replace("\n", " ")
        print("  [" + c["chapter_id"] + "] " + c["topic_slug"] + " | " + preview)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(_json.dumps(chunk, ensure_ascii=False) + "\n")
    print("Written", len(chunks), "chunks to", OUT_JSONL)


if __name__ == "__main__":
    main()
