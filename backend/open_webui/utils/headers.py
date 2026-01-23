from urllib.parse import quote


def include_user_info_headers(headers, user):
    return {
        **headers,
        "X-Gnos3-User-Name": quote(user.name, safe=" "),
        "X-Gnos3-User-Id": user.id,
        "X-Gnos3-User-Email": user.email,
        "X-Gnos3-User-Role": user.role,
    }
