from common.minio_base import BaseMinIOClient


class MinIOClient(BaseMinIOClient):
    def __init__(self):
        super().__init__("prompts")

    def upload_prompt(self, content: str, prompt_id: str, version_number: int) -> str:
        """Upload prompt content to MinIO and return the key"""
        key = f"prompts/{prompt_id}/v{version_number}"
        return self.upload_content(content, key, 'text/plain')

    def download_prompt(self, key: str) -> str:
        """Download prompt content from MinIO"""
        return self.download_content(key)

    def delete_prompt(self, key: str):
        """Delete prompt from MinIO"""
        self.delete_content(key)


# Global instance
minio_client = MinIOClient()