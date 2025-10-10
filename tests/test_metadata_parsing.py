import textwrap

from data_extraction import (
    assemble_document_extraction,
    extract_case_numbers,
    extract_case_types,
    extract_dates,
    extract_judgments,
    extract_party_names,
    ordered_unique,
    PartyMetadata,
)
from pathlib import Path


def test_ordered_unique_preserves_order():
    assert ordered_unique(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_extract_case_numbers_varied_patterns():
    sample = textwrap.dedent(
        """
        Criminal Appeal No. 12 of 2023
        Writ Petition No: 45 of 2021
        Case Number: 88/2022
        Review Case No- 5 of 2019
        Misc. Case No. 123/2020
        """
    )
    case_numbers = extract_case_numbers(sample)
    assert "12 of 2023" in case_numbers
    assert "45 of 2021" in case_numbers
    assert "88/2022" in case_numbers
    assert "5 of 2019" in case_numbers
    assert "123/2020" in case_numbers


def test_extract_case_types_recognises_common_labels():
    sample = textwrap.dedent(
        """
        Criminal Appeal No. 12 of 2023
        Review Case Number 4 of 2020
        ফৌজদারি আপিল নং ৩ of 2018
        Writ Petition No: 45 of 2021
        Misc. Case No. 123/2020
        """
    )
    case_types = extract_case_types(sample)
    assert "Criminal Appeal" in case_types
    assert "Review Case" in case_types
    assert "ফৌজদারি আপিল" in case_types
    assert "Writ Petition" in case_types
    assert "Misc. Case" in case_types


def test_extract_dates_numeric_and_textual():
    sample = textwrap.dedent(
        """
        Judgment delivered on 14/02/2023.
        Order dated 1st January, 2021.
        Filing date: 07-03-19.
        """
    )
    dates = extract_dates(sample)
    assert "2023-02-14" in dates
    assert "2021-01-01" in dates
    assert "2019-03-07" in dates


def test_extract_party_names_roles_and_versus():
    sample = textwrap.dedent(
        """
        Md. Rahim vs The State
        Petitioner: Md. Rahim
        Respondent: The State
        Appellant: Company XYZ
        Opposite Party: জন Doe
        """
    )
    parties = extract_party_names(sample)
    assert ("Md. Rahim", "The State") in parties.versus_pairs
    assert parties.role_based["petitioners"] == ["Md. Rahim"]
    assert parties.role_based["respondents"] == ["The State"]
    assert parties.role_based["appellants"] == ["Company XYZ"]
    assert parties.role_based["opposite_parties"] == ["জন Doe"]


def test_extract_judgments_captures_outcomes():
    sample = textwrap.dedent(
        """
        The appeal is allowed with costs.
        Petition dismissed for default.
        Judgment set aside.
        Bail granted to the petitioner.
        """
    )
    judgments = extract_judgments(sample)
    assert "Appeal allowed" in judgments
    assert "Petition dismissed" in judgments
    assert "Judgment set aside" in judgments
    assert "Bail granted" in judgments


def test_assemble_document_extraction_structure(tmp_path: Path):
    text = textwrap.dedent(
        """
        Civil Petition No. 123 of 2020
        Judgment Date: 05/11/2021
        Rahima Khatun v. Bangladesh Bank
        Petitioner: Rahima Khatun
        Respondent: Bangladesh Bank
        The petition is dismissed.
        """
    )
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test content")

    extraction = assemble_document_extraction(pdf_path, text)

    assert extraction.source_pdf == pdf_path
    assert "123 of 2020" in extraction.case_numbers
    assert "Civil Petition" in extraction.case_types
    assert "2021-11-05" in extraction.dates
    assert ("Rahima Khatun", "Bangladesh Bank") in extraction.party_metadata.versus_pairs
    assert extraction.party_metadata.role_based["petitioners"] == ["Rahima Khatun"]
    assert extraction.party_metadata.role_based["respondents"] == ["Bangladesh Bank"]
    assert "Petition dismissed" in extraction.judgments
