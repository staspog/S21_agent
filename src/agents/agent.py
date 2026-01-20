"""LangGraph агент для обработки запросов пользователей"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from ..llm import LLMClient
from ..prompts import load_prompt, format_prompt
from ..database import MockProjectDatabase
from ..integrations import RocketChatClient
from ..config import MAX_PROJECT_SUGGESTIONS, PROJECT_CONFIRMATION_REQUIRED


class AgentState(TypedDict):
    """Состояние агента"""
    user_query: str
    query_type: Literal["ТЕХНИЧЕСКИЙ", "ОРГАНИЗАЦИОННЫЙ", None]
    project_names: list[str]
    project_context: str
    search_query: str
    answer: str


class S21Agent:
    """LangGraph агент для обработки запросов"""
    
    def __init__(self):
        """Инициализация агента"""
        self.llm = LLMClient()
        self.db = MockProjectDatabase()
        self.rocketchat = RocketChatClient()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Строит граф агента"""
        graph = StateGraph(AgentState)
        
        # Добавляем узлы
        graph.add_node("classify", self._classify_query)
        graph.add_node("find_projects", self._find_projects)
        graph.add_node("confirm_projects", self._confirm_projects)
        graph.add_node("get_technical_answer", self._get_technical_answer)
        graph.add_node("get_organizational_answer", self._get_organizational_answer)
        
        # Устанавливаем входную точку
        graph.set_entry_point("classify")
        
        # Добавляем conditional edge после классификации
        graph.add_conditional_edges(
            "classify",
            self._route_by_type,
            {
                "ТЕХНИЧЕСКИЙ": "find_projects",
                "ОРГАНИЗАЦИОННЫЙ": "get_organizational_answer",
            }
        )
        
        # Маршрутизация после поиска проектов
        graph.add_conditional_edges(
            "find_projects",
            self._route_after_project_search,
            {
                "confirm": "confirm_projects",
                "answer": "get_technical_answer",
            }
        )
        
        # После подтверждения проектов идем к ответу
        graph.add_edge("confirm_projects", "get_technical_answer")
        
        # Финальные узлы ведут к END
        graph.add_edge("get_technical_answer", END)
        graph.add_edge("get_organizational_answer", END)
        
        return graph.compile()
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Классифицирует запрос пользователя"""
        classification_prompt = load_prompt("classification.txt")
        
        full_prompt = f"{classification_prompt}\n\nЗапрос пользователя: {state['user_query']}"
        
        query_type = self.llm.invoke_structured(
            full_prompt,
            response_format="одно слово: ТЕХНИЧЕСКИЙ или ОРГАНИЗАЦИОННЫЙ"
        ).strip()
        
        # Нормализуем ответ
        if "ТЕХНИЧЕСКИЙ" in query_type.upper():
            query_type = "ТЕХНИЧЕСКИЙ"
        elif "ОРГАНИЗАЦИОННЫЙ" in query_type.upper():
            query_type = "ОРГАНИЗАЦИОННЫЙ"
        else:
            # По умолчанию считаем техническим, если не понятно
            query_type = "ТЕХНИЧЕСКИЙ"
        
        state["query_type"] = query_type
        return state
    
    def _route_by_type(self, state: AgentState) -> str:
        """Маршрутизация по типу запроса"""
        return state["query_type"]
    
    def _find_projects(self, state: AgentState) -> AgentState:
        """Находит релевантные проекты"""
        # Получаем список всех проектов
        all_projects = self.db.get_all_projects()
        
        # Формируем список для промпта
        projects_list = "\n".join([
            f"- {proj['name']}: {proj['description']}"
            for proj in all_projects
        ])
        
        # Загружаем промпт для поиска проектов
        project_search_prompt = load_prompt("project_search.txt")
        formatted_prompt = format_prompt(
            project_search_prompt,
            projects_list=projects_list
        )
        
        full_prompt = f"{formatted_prompt}\n\nЗапрос пользователя: {state['user_query']}"
        
        # Получаем ответ от LLM
        response = self.llm.invoke(full_prompt).strip()
        
        # Парсим список проектов
        if "НЕИЗВЕСТНО" in response.upper():
            project_names = []
        else:
            # Разделяем по запятой и очищаем
            project_names = [
                name.strip() 
                for name in response.split(",")
                if name.strip()
            ]
            # Ограничиваем количество
            project_names = project_names[:MAX_PROJECT_SUGGESTIONS]
        
        state["project_names"] = project_names
        return state
    
    def _route_after_project_search(self, state: AgentState) -> str:
        """Маршрутизация после поиска проектов"""
        if not state["project_names"]:
            # Если проекты не найдены, сразу отвечаем
            return "answer"
        
        if PROJECT_CONFIRMATION_REQUIRED and len(state["project_names"]) > 1:
            # Если требуется подтверждение и найдено несколько проектов
            return "confirm"
        
        # Иначе сразу отвечаем
        return "answer"
    
    def _confirm_projects(self, state: AgentState) -> AgentState:
        """Подтверждение выбора проектов (мок - в реальности здесь будет интерактив)"""
        # В реальной реализации здесь будет запрос подтверждения у пользователя
        # Пока что просто используем первый проект или все найденные
        
        if not state["project_names"]:
            state["project_names"] = []
            return state
        
        # Для демо используем все найденные проекты
        # В реальности здесь будет логика подтверждения
        return state
    
    def _get_technical_answer(self, state: AgentState) -> AgentState:
        """Получает ответ на технический вопрос"""
        if not state["project_names"]:
            state["answer"] = "Не удалось определить проект по вашему запросу. Пожалуйста, уточните название проекта."
            return state
        
        # Собираем контекст из всех найденных проектов
        project_contexts = []
        for project_name in state["project_names"]:
            tasks = self.db.get_project_tasks(project_name)
            if tasks:
                project_contexts.append(f"## Проект: {project_name}\n\n{tasks}")
        
        if not project_contexts:
            state["answer"] = f"Не найдены задания для проектов: {', '.join(state['project_names'])}"
            return state
        
        # Объединяем контексты
        full_context = "\n\n".join(project_contexts)
        
        # Загружаем промпт для технического ответа
        technical_prompt = load_prompt("technical_answer.txt")
        formatted_prompt = format_prompt(
            technical_prompt,
            project_context=full_context,
            user_query=state["user_query"]
        )
        
        # Получаем ответ от LLM
        answer = self.llm.invoke(formatted_prompt)
        state["answer"] = answer
        return state
    
    def _get_organizational_answer(self, state: AgentState) -> AgentState:
        """Получает ответ на организационный вопрос"""
        # Загружаем промпт для формирования поискового запроса
        search_prompt_template = load_prompt("rocketchat_search.txt")
        search_prompt = format_prompt(
            search_prompt_template,
            user_query=state["user_query"]
        )
        
        # Получаем поисковый запрос от LLM
        search_query = self.llm.invoke(search_prompt).strip()
        state["search_query"] = search_query
        
        # Ищем в RocketChat
        search_results = self.rocketchat.search(search_query)
        formatted_results = self.rocketchat.format_search_results(search_results)
        
        # Формируем финальный ответ
        answer = f"По вашему организационному вопросу:\n\n{formatted_results}"
        state["answer"] = answer
        return state
    
    def invoke(self, user_query: str) -> str:
        """
        Обрабатывает запрос пользователя
        
        Args:
            user_query: Запрос пользователя
            
        Returns:
            Ответ агента
        """
        initial_state: AgentState = {
            "user_query": user_query,
            "query_type": None,
            "project_names": [],
            "project_context": "",
            "search_query": "",
            "answer": "",
        }
        
        # Запускаем граф
        final_state = self.graph.invoke(initial_state)
        
        return final_state["answer"]
