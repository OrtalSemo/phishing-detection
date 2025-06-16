import os
import pprint

def get_user_role():
    """
    Determine the current user's role.
    For now, use an environment variable or default to 'user'.
    """
    return os.getenv("SAFENET_ROLE", "user").lower()  # 'user' or 'admin'

def filter_event(event: dict, role: str = None) -> dict:
    """
    Filter event data based on the role.
    :param event: dict with all event data (phishing detection result)
    :param role: 'user' or 'admin'
    :return: filtered dict
    """
    if role is None:
        role = get_user_role()

    if role == "admin":
        # Show everything
        return event

    # User mode: show only minimal/essential info
    filtered = {
        "timestamp": event.get("timestamp"),
        "phishing_status": event.get("phishing_status"),
        "suspicious_urls": event.get("suspicious_urls"),
        "summary": event.get("summary"),
    }
    return filtered

def format_event(event: dict, role: str = None) -> str:
    """
    Format the event for display/export based on role.
    """
    filtered = filter_event(event, role)
    if get_user_role() == "admin":
        # Pretty print all info for admin
        return pprint.pformat(filtered)
    else:
        # User: concise summary
        lines = [
            f"Time: {filtered.get('timestamp')}",
            f"Status: {filtered.get('phishing_status')}",
        ]
        if filtered.get("suspicious_urls"):
            lines.append(f"Suspicious URLs: {filtered['suspicious_urls']}")
        if filtered.get("summary"):
            lines.append(f"Summary: {filtered['summary']}")
        return "\n".join(lines)

def export_for_splunk(event: dict, role: str = None) -> dict:
    """
    Return only the fields that should be sent to Splunk, according to permissions.
    """
    filtered = filter_event(event, role)
    # You can further customize which fields are sent to Splunk here if needed
    return filtered
