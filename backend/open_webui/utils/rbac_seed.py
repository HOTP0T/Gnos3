"""
Cross-stack RBAC: seed the four default groups on Gnos3 startup.

Idempotent — uses get-or-create semantics keyed by group name. Existing groups
are NOT overwritten (admins may have customized them). To re-apply the canonical
permission tree, delete the group via the admin UI and restart.

Roles seeded:
  - Admins              full access to every module + accounting admin
  - Accountants         invoices read; accounting read/write/post; companies "*": accountant
  - Document Operators  invoices + BC read/write/reprocess; K4mi docs read/write
  - Viewers             read on every module; K4mi docs read

The `companies` block under each group's permissions uses `"*"` to mean "every
company". Phase 3 will introduce per-company overrides via a CompanyMembership
table; until then, the "*" key alone is honored by the modules.
"""

import logging
from typing import Any

from open_webui.models.groups import GroupForm, Groups
from open_webui.models.users import Users

log = logging.getLogger(__name__)

# Permission tree for each canonical role. Mirrors the structure under
# DEFAULT_USER_PERMISSIONS["modules"] / ["k4mi"] / ["companies"].
_GROUP_PERMS: dict[str, dict[str, Any]] = {
    'Admins': {
        'modules': {
            'invoices':       {'read': True, 'write': True, 'reprocess': True, 'delete': True},
            'business_cards': {'read': True, 'write': True, 'reprocess': True, 'delete': True},
            'accounting':     {'read': True, 'write': True, 'post': True, 'admin': True},
        },
        'companies': {'*': {'role': 'admin'}},
        'k4mi': {'documents': {'read': True, 'write': True, 'admin': True}},
    },
    'Accountants': {
        'modules': {
            'invoices':       {'read': True, 'write': False, 'reprocess': False, 'delete': False},
            'business_cards': {'read': False, 'write': False, 'reprocess': False, 'delete': False},
            'accounting':     {'read': True, 'write': True, 'post': True, 'admin': False},
        },
        'companies': {'*': {'role': 'accountant'}},
        'k4mi': {'documents': {'read': True, 'write': False, 'admin': False}},
    },
    'Document Operators': {
        'modules': {
            'invoices':       {'read': True, 'write': True, 'reprocess': True, 'delete': False},
            'business_cards': {'read': True, 'write': True, 'reprocess': True, 'delete': False},
            'accounting':     {'read': False, 'write': False, 'post': False, 'admin': False},
        },
        'companies': {'*': {'role': 'viewer'}},
        'k4mi': {'documents': {'read': True, 'write': True, 'admin': False}},
    },
    'Viewers': {
        'modules': {
            'invoices':       {'read': True, 'write': False, 'reprocess': False, 'delete': False},
            'business_cards': {'read': True, 'write': False, 'reprocess': False, 'delete': False},
            'accounting':     {'read': True, 'write': False, 'post': False, 'admin': False},
        },
        'companies': {'*': {'role': 'viewer'}},
        'k4mi': {'documents': {'read': True, 'write': False, 'admin': False}},
    },
}

_GROUP_DESCRIPTIONS = {
    'Admins': 'Full access to every data module, all companies, and all K4mi documents.',
    'Accountants': (
        'Read invoices; read/write/post accounting entries. Cannot delete companies, '
        'manage chart of accounts, or change permissions.'
    ),
    'Document Operators': (
        'Read/write invoices and business cards (including reprocess). Can edit K4mi '
        'documents. No accounting access.'
    ),
    'Viewers': 'Read-only across every data module. No mutation, no posting, no admin.',
}


async def ensure_default_rbac_groups() -> None:
    """Create the four canonical RBAC groups if they don't already exist.

    Uses the first user (Open WebUI's auto-admin) as the `user_id` owner of
    newly-created groups (Group rows require a creator). If the DB has no
    users yet (very-first boot, no signup), silently defers — the next
    restart picks it up once an admin signs up.
    """
    owner = None
    try:
        owner = await Users.get_first_user()
    except Exception:
        log.debug('RBAC seed: Users.get_first_user failed; deferring to next restart.')
        return

    if owner is None:
        log.info(
            'RBAC seed: no users yet — default groups will be created on the '
            'next restart after the first user signs up.'
        )
        return

    admin_user_id = owner.id

    created = 0
    for name, permissions in _GROUP_PERMS.items():
        try:
            existing = await Groups.get_group_by_name(name)
            if existing is not None:
                continue
            await Groups.insert_new_group(
                user_id=admin_user_id,
                form_data=GroupForm(
                    name=name,
                    description=_GROUP_DESCRIPTIONS[name],
                    permissions=permissions,
                ),
            )
            created += 1
            log.info(f'RBAC seed: created group {name!r}')
        except Exception as exc:
            log.warning(f'RBAC seed: failed to create group {name!r}: {exc}')

    if created:
        log.info(f'RBAC seed: {created} default group(s) created. Existing groups left untouched.')
    else:
        log.debug('RBAC seed: all default groups already present.')
