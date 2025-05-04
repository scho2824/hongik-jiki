"""
LangChain 통합을 위한 LLM 구현
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from hongikjiki.langchain_integration.base import LLMBase

logger = logging.getLogger("HongikJikiChatBot")

class OpenAILLM(LLMBase):
    """OpenAI API를 사용한 LLM 구현"""
    
    def __init__(self, 
                 model: str = "gpt-4o", 
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        OpenAI LLM 초기화
        
        Args:
            model: 사용할 OpenAI 모델명
            api_key: OpenAI API 키
            temperature: 생성 온도 (창의성 조절)
            max_tokens: 최대 생성 토큰 수
        """
        try:
            # OpenAI 라이브러리 임포트
            import openai
            
            # API 키 설정
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API 키가 필요합니다. API 키를 직접 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
            
            # 모델 및 설정
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            # OpenAI 클라이언트 초기화
            self.client = openai.OpenAI(api_key=self.api_key)
            if model != "gpt-4o":
                logger.warning(f"⚠️ 현재 모델은 '{model}'입니다. 'gpt-4o'가 아닌 값이 사용되고 있습니다.")
            logger.info(f"OpenAI LLM 초기화 완료: 모델={model}, 온도={temperature}")
            
        except ImportError:
            logger.error("openai 라이브러리를 찾을 수 없습니다. 'pip install openai'를 실행하세요.")
            raise
        except Exception as e:
            logger.error(f"OpenAI LLM 초기화 오류: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        단일 프롬프트로부터 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 매개변수 (temperature, max_tokens 등 재정의 가능)
            
        Returns:
            str: 생성된 텍스트
        """
        try:
            # 기본 설정에 추가 매개변수 병합
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Chat Completion API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 응답 추출
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            return f"[오류 발생: {str(e)}]"
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        여러 프롬프트로부터 텍스트 일괄 생성
        
        Args:
            prompts: 입력 프롬프트 리스트
            **kwargs: 추가 매개변수
            
        Returns:
            List[str]: 생성된 텍스트 리스트
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

    def generate_text(self, prompt: str) -> str:
        """
        주어진 프롬프트에 대한 텍스트 생성

        Args:
            prompt: 프롬프트 텍스트

        Returns:
            str: 생성된 텍스트
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            raise

    def test_completion(self, prompt: str) -> str:
        """
        OpenAI API 호출 테스트

        Args:
            prompt: 테스트할 프롬프트

        Returns:
            str: 생성된 응답
        """
        try:
            # API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 응답 추출
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API 호출 오류: {e}")
            return f"오류 발생: {str(e)}"
    
class NaverClovaLLM(LLMBase):
    """네이버 Clova API를 사용한 LLM 구현"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_gateway: str = "https://clovastudio.apigw.ntruss.com/testapp/v1/completions/LK-D",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Clova LLM 초기화
        
        Args:
            api_key: Clova API 키
            api_gateway: API 게이트웨이 주소
            temperature: 생성 온도 (창의성 조절)
            max_tokens: 최대 생성 토큰 수
        """
        try:
            # requests 라이브러리 임포트
            import requests
            
            # API 키 설정
            self.api_key = api_key or os.environ.get("CLOVA_API_KEY")
            if not self.api_key:
                raise ValueError("Clova API 키가 필요합니다. API 키를 직접 전달하거나 CLOVA_API_KEY 환경 변수를 설정하세요.")
            
            # 모델 및 설정
            self.api_gateway = api_gateway
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.requests = requests
            
            logger.info(f"Naver Clova LLM 초기화 완료: 온도={temperature}")
            
        except ImportError:
            logger.error("requests 라이브러리를 찾을 수 없습니다. 'pip install requests'를 실행하세요.")
            raise
        except Exception as e:
            logger.error(f"Clova LLM 초기화 오류: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        단일 프롬프트로부터 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 매개변수
            
        Returns:
            str: 생성된 텍스트
        """
        try:
            # 기본 설정에 추가 매개변수 병합
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # API 요청 데이터 구성
            request_data = {
                "text": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.8),
                "top_k": kwargs.get("top_k", 0),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "n": 1,
                "stop": kwargs.get("stop", [])
            }
            
            # 요청 헤더
            headers = {
                "Content-Type": "application/json",
                "X-NCP-CLOVASTUDIO-API-KEY": self.api_key
            }
            
            # API 호출
            response = self.requests.post(
                self.api_gateway,
                json=request_data,
                headers=headers
            )
            
            # 응답 파싱
            if response.status_code == 200:
                result = response.json()
                return result.get("result", {}).get("text", "").strip()
            else:
                logger.error(f"Clova API 오류: {response.status_code} {response.text}")
                return f"[API 오류: {response.status_code}]"
        
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            return f"[오류 발생: {str(e)}]"
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        여러 프롬프트로부터 텍스트 일괄 생성
        
        Args:
            prompts: 입력 프롬프트 리스트
            **kwargs: 추가 매개변수
            
        Returns:
            List[str]: 생성된 텍스트 리스트
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results


def get_llm(llm_type: str = "openai", **kwargs) -> LLMBase:
    """
    LLM 타입에 따른 적절한 LLM 인스턴스 생성
    
    Args:
        llm_type: LLM 타입 ('openai', 'clova' 등)
        **kwargs: 선택한 LLM 클래스에 전달할 추가 인자
        
    Returns:
        LLMBase: LLM 인스턴스
    """
    llm_types = {
        "openai": OpenAILLM,
        "clova": NaverClovaLLM
    }
    
    if llm_type not in llm_types:
        logger.warning(f"지원하지 않는 LLM 타입: {llm_type}, 기본값(openai) 사용")
        llm_type = "openai"

    # 모델 강제 지정 (gpt-4o를 최우선으로 사용)
    if "model" not in kwargs or not kwargs["model"]:
        kwargs["model"] = "gpt-4o"
    
    # 적절한 LLM 클래스 생성 및 반환
    return llm_types[llm_type](**kwargs)