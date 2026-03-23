from typing import List

from pydantic import BaseModel


class FAQ(BaseModel):
    question: str
    answer: str


class BulkFAQ(BaseModel):
    faqs: List[FAQ]


class Query(BaseModel):
    query: str
