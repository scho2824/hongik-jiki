from typing import Dict, Any, Optional, List, Union,Tuple
import logging

logger = logging.getLogger("HongikJikiChatBot")

class EmbeddingsBase:
    """임베딩 모델 기본 인터페이스"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 텍스트를 벡터로 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 텍스트를 벡터로 임베딩
        
        Args:
            text: 임베딩할, 쿼리 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")


class HuggingFaceEmbeddings(EmbeddingsBase):
    """
    HuggingFace 모델을 사용한 임베딩 구현
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", **kwargs):
        """
        HuggingFaceEmbeddings 초기화
        
        Args:
            model_name: HuggingFace 모델 이름
            kwargs: 추가 매개변수
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, **kwargs)
            logger.info(f"HuggingFace 임베딩 모델 로드: {model_name}")
        except ImportError:
            logger.error("sentence-transformers 패키지가 설치되지 않았습니다.")
            raise ImportError("sentence-transformers 패키지를 설치하세요: pip install sentence-transformers")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 텍스트를 벡터로 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 텍스트를 벡터로 임베딩
        
        Args:
            text: 임베딩할 쿼리 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
        """
        embedding = self.model.encode(text)
        return embedding.tolist()


class OpenAIEmbeddings(EmbeddingsBase):
    """
    OpenAI API를 사용한 임베딩 구현
    """
    
    def __init__(self, model: str = "text-embedding-3-small", **kwargs):
        """
        OpenAIEmbeddings 초기화
        
        Args:
            model: OpenAI 임베딩 모델 이름
            kwargs: 추가 매개변수
        """
        try:
            import openai
            self.client = openai.OpenAI()
            self.model = model
            self.kwargs = kwargs
            logger.info(f"OpenAI 임베딩 모델 설정: {model}")
        except ImportError:
            logger.error("openai 패키지가 설치되지 않았습니다.")
            raise ImportError("openai 패키지를 설치하세요: pip install openai")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 텍스트를 벡터로 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        # 청크가 큰 경우 분할하여 처리
        embeddings = []
        for i in range(0, len(texts), 20):  # API 제한 고려
            batch = texts[i:i+20]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                **self.kwargs
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 텍스트를 벡터로 임베딩
        
        Args:
            text: 임베딩할 쿼리 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            **self.kwargs
        )
        return response.data[0].embedding


def get_embeddings(embedding_type: str = "huggingface", **kwargs) -> EmbeddingsBase:
    """
    임베딩 타입에 따른 임베딩 객체 생성
    
    Args:
        embedding_type: 임베딩 타입 ("huggingface" 또는 "openai")
        kwargs: 임베딩 생성자에 전달할 매개변수
        
    Returns:
        EmbeddingsBase: 임베딩 객체
    """
    if embedding_type.lower() == "huggingface":
        return HuggingFaceEmbeddings(**kwargs)
    elif embedding_type.lower() == "openai":
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 임베딩 타입: {embedding_type}")