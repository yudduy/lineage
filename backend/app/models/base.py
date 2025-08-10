"""
Base Pydantic model with common configurations.
"""

from typing import Any, Dict
from pydantic import BaseModel as PydanticBaseModel, Field
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    """Base model with common configuration for all models."""
    
    model_config = ConfigDict(
        # Allow population by field name and alias
        populate_by_name=True,
        # Validate default values
        validate_default=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Allow extra fields for flexibility
        extra="ignore",
        # Serialize by alias
        ser_by_alias=True,
    )
    
    def dict_without_none(self) -> Dict[str, Any]:
        """Return dictionary representation excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
    
    def dict_for_db(self) -> Dict[str, Any]:
        """Return dictionary representation suitable for database storage."""
        return self.model_dump(exclude_unset=True, by_alias=True)