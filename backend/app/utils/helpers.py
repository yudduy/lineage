"""
Utility helper functions.
"""

import re
import uuid
from typing import Any, Dict, Optional
from datetime import datetime


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def validate_doi(doi: str) -> bool:
    """
    Validate DOI format.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        True if valid DOI format, False otherwise
    """
    if not doi or not isinstance(doi, str):
        return False
    
    # DOI must start with "10."
    if not doi.startswith("10."):
        return False
    
    # Basic DOI pattern validation
    doi_pattern = r'^10\.\d{4,}/[^\s]+$'
    return bool(re.match(doi_pattern, doi))


def normalize_doi(doi: str) -> Optional[str]:
    """
    Normalize DOI format.
    
    Args:
        doi: DOI string to normalize
        
    Returns:
        Normalized DOI or None if invalid
    """
    if not doi:
        return None
    
    # Remove common prefixes
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[16:]
    elif doi.startswith("http://doi.org/"):
        doi = doi[15:]
    elif doi.startswith("doi:"):
        doi = doi[4:]
    
    # Validate and return
    if validate_doi(doi):
        return doi
    
    return None


def format_response(
    success: bool = True,
    message: str = "",
    data: Any = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format standard API response.
    
    Args:
        success: Whether the operation was successful
        message: Response message
        data: Response data
        meta: Additional metadata
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if data is not None:
        response["data"] = data
    
    if meta is not None:
        response["meta"] = meta
    
    return response


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for file system
    """
    if not filename:
        return "untitled"
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 250 - len(ext)
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    # Ensure it's not empty
    if not filename.strip():
        return "untitled"
    
    return filename.strip()


def extract_paper_id_from_url(url: str) -> Optional[str]:
    """
    Extract paper ID from various URL formats.
    
    Args:
        url: URL containing paper identifier
        
    Returns:
        Extracted paper ID or None if not found
    """
    if not url:
        return None
    
    # DOI URLs
    doi_patterns = [
        r'doi\.org/(.+)$',
        r'dx\.doi\.org/(.+)$',
        r'/doi/(.+)$'
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, url)
        if match:
            doi = match.group(1)
            if validate_doi(f"10.{doi}" if not doi.startswith("10.") else doi):
                return doi
    
    # ArXiv URLs
    arxiv_pattern = r'arxiv\.org/abs/(.+)$'
    match = re.search(arxiv_pattern, url)
    if match:
        return match.group(1)
    
    # PubMed URLs
    pubmed_pattern = r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)'
    match = re.search(pubmed_pattern, url)
    if match:
        return match.group(1)
    
    return None


def format_author_name(first_name: str, last_name: str) -> str:
    """
    Format author name consistently.
    
    Args:
        first_name: Author's first name
        last_name: Author's last name
        
    Returns:
        Formatted author name
    """
    if not first_name and not last_name:
        return "Unknown Author"
    
    if not first_name:
        return last_name.strip()
    
    if not last_name:
        return first_name.strip()
    
    return f"{first_name.strip()} {last_name.strip()}"


def parse_citation_string(citation: str) -> Dict[str, str]:
    """
    Parse citation string into components.
    
    Args:
        citation: Citation string to parse
        
    Returns:
        Dictionary with parsed citation components
    """
    # This is a simplified parser - in production you'd use a more sophisticated approach
    result = {
        "title": "",
        "authors": "",
        "journal": "",
        "year": "",
        "volume": "",
        "pages": ""
    }
    
    if not citation:
        return result
    
    # Try to extract year
    year_match = re.search(r'\((\d{4})\)', citation)
    if year_match:
        result["year"] = year_match.group(1)
    
    # Try to extract title (usually in quotes)
    title_match = re.search(r'"([^"]+)"', citation)
    if title_match:
        result["title"] = title_match.group(1)
    
    return result


def calculate_citation_metrics(citations: list) -> Dict[str, Any]:
    """
    Calculate citation metrics from citation list.
    
    Args:
        citations: List of citation records
        
    Returns:
        Dictionary with calculated metrics
    """
    if not citations:
        return {
            "total_citations": 0,
            "h_index": 0,
            "citations_by_year": {},
            "average_citations_per_paper": 0
        }
    
    # Count citations by year
    citations_by_year = {}
    total_citations = len(citations)
    
    for citation in citations:
        year = citation.get("year", "unknown")
        citations_by_year[year] = citations_by_year.get(year, 0) + 1
    
    # Calculate h-index (simplified)
    citation_counts = sorted([len(citations)] * len(set(c.get("paper_id") for c in citations)), reverse=True)
    h_index = 0
    for i, count in enumerate(citation_counts):
        if count >= i + 1:
            h_index = i + 1
        else:
            break
    
    return {
        "total_citations": total_citations,
        "h_index": h_index,
        "citations_by_year": citations_by_year,
        "average_citations_per_paper": total_citations / len(set(c.get("paper_id") for c in citations)) if citations else 0
    }


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)].rstrip()
    return truncated + suffix


def clean_html(html_text: str) -> str:
    """
    Clean HTML tags from text.
    
    Args:
        html_text: HTML text to clean
        
    Returns:
        Plain text with HTML tags removed
    """
    if not html_text:
        return ""
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    
    # Decode common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
    }
    
    for entity, char in html_entities.items():
        clean_text = clean_text.replace(entity, char)
    
    return clean_text.strip()


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    if not url:
        return False
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data for logging.
    
    Args:
        data: Sensitive data to mask
        visible_chars: Number of characters to show at the end
        
    Returns:
        Masked data string
    """
    if not data or len(data) <= visible_chars:
        return "*" * len(data) if data else ""
    
    return "*" * (len(data) - visible_chars) + data[-visible_chars:]