"""Password hashing (argon2id) and the password policy."""

from __future__ import annotations

import re
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

MIN_LENGTH = 12
MAX_LENGTH = 128  # also stops very long inputs being used to make the server do slow hashing work

_hasher = PasswordHasher()  # argon2id with the library's default cost settings

# A short built-in list, not a full breached-password list. Only entries long enough to pass the length rule
# matter here. Add a larger list (or a check against a breach service) before relying on this.
_COMMON = frozenset(
    """
123456789012 1234567890123 12345678901234 123456789abc 1q2w3e4r5t6y 1qaz2wsx3edc abcdefghijkl
administrator changemenow1 correcthorse letmein12345 letmeinplease passw0rd1234 password1234 password12345
password123456 passwordpassword qwertyuiop12 qwertyuiopasdf qwerty123456 qwertyqwerty welcome12345 welcometothejungle
iloveyou1234 iloveyou12345 monkey123456 football1234 baseball1234 superman1234 trustno1trustno1 abc123abc123
zaq12wsxcde3 asdfghjkl123 asdfghjklqwerty thequickbrownfox dragon123456 master123456 sunshine1234 princess1234
""".split()
)


class PasswordPolicyError(ValueError):
    """The password is not acceptable. The message is safe to show to the user."""


def check_policy(password: str, email: Optional[str] = None) -> None:
    if len(password) < MIN_LENGTH:
        raise PasswordPolicyError(
            f"Use at least {MIN_LENGTH} characters. A few words with spaces or dashes works well."
        )
    if len(password) > MAX_LENGTH:
        raise PasswordPolicyError(f"Use at most {MAX_LENGTH} characters.")
    lowered = password.lower()
    if lowered in _COMMON or re.fullmatch(r"(.)\1+", password):
        raise PasswordPolicyError("That password is too common. Choose something less guessable.")
    if email:
        local = email.split("@")[0].lower()
        if len(local) >= 4 and lowered.replace(" ", "") == local:
            raise PasswordPolicyError("The password should not be your email address.")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored: str) -> bool:
    try:
        return _hasher.verify(stored, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(stored: str) -> bool:
    return _hasher.check_needs_rehash(stored)


# Verified when the email does not exist, so a missing account takes as long as a wrong password.
_DUMMY_HASH = _hasher.hash("not-a-real-password-for-timing")


def burn_time() -> None:
    verify_password("irrelevant", _DUMMY_HASH)
