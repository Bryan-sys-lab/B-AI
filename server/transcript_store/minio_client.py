from common.minio_base import BaseMinIOClient


class MinIOClient(BaseMinIOClient):
    def __init__(self):
        super().__init__("transcripts")

    def upload_transcript(self, content: str, transcript_id: str) -> str:
        """Upload transcript content to MinIO and return the key"""
        key = f"transcripts/{transcript_id}"
        return self.upload_content(content, key, 'application/json')

    def download_transcript(self, key: str) -> str:
        """Download transcript content from MinIO"""
        return self.download_content(key)

    def delete_transcript(self, key: str):
        """Delete transcript from MinIO"""
        self.delete_content(key)


# Global instance
minio_client = MinIOClient()