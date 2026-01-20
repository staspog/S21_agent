"""LangGraph агент для обработки запросов пользователей"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import json
import time

from ..llm import LLMClient
from ..prompts import load_prompt, format_prompt
from ..database import MockProjectDatabase
from ..integrations import RocketChatClient
from ..config import MAX_PROJECT_SUGGESTIONS, PROJECT_CONFIRMATION_REQUIRED
from ..utils import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Состояние агента"""
    user_query: str
    query_type: Literal["ТЕХНИЧЕСКИЙ", "ОРГАНИЗАЦИОННЫЙ", None]
    project_names: list[str]
    project_context: str
    search_query: str
    answer: str
    # Для уточняющих вопросов
    collected_info: dict  # Собранная информация (group, language, order, etc.)
    questions_asked: int  # Количество заданных вопросов
    current_question: str  # Текущий вопрос для пользователя
    needs_user_input: bool  # Нужен ли ответ от пользователя
    session_id: str  # ID сессии для логирования


class S21Agent:
    """LangGraph агент для обработки запросов"""
    
    def __init__(self):
        """Инициализация агента"""
        logger.info("Инициализация S21Agent")
        start_time = time.time()
        
        self.llm = LLMClient()
        logger.debug("LLM клиент инициализирован")
        
        self.db = MockProjectDatabase()
        logger.debug("База данных проектов инициализирована")
        
        self.rocketchat = RocketChatClient()
        logger.debug("RocketChat клиент инициализирован")
        
        self.graph = self._build_graph()
        logger.info(f"Граф агента построен за {time.time() - start_time:.3f}s")
    
    def _build_graph(self) -> StateGraph:
        """Строит граф агента"""
        graph = StateGraph(AgentState)
        
        # Добавляем узлы
        graph.add_node("classify", self._classify_query)
        graph.add_node("plan_project_search", self._plan_project_search)
        graph.add_node("ask_clarification_question", self._ask_clarification_question)
        graph.add_node("filter_projects", self._filter_projects)
        graph.add_node("get_technical_answer", self._get_technical_answer)
        graph.add_node("get_organizational_answer", self._get_organizational_answer)
        
        # Устанавливаем входную точку
        graph.set_entry_point("classify")
        
        # Добавляем conditional edge после классификации
        graph.add_conditional_edges(
            "classify",
            self._route_by_type,
            {
                "ТЕХНИЧЕСКИЙ": "plan_project_search",
                "ОРГАНИЗАЦИОННЫЙ": "get_organizational_answer",
            }
        )
        
        # Маршрутизация после планирования поиска
        graph.add_conditional_edges(
            "plan_project_search",
            self._route_after_planning,
            {
                "search": "filter_projects",
                "ask_question": "ask_clarification_question",
                "insufficient": "get_technical_answer",  # Сообщение о недостаточности информации
            }
        )
        
        # После вопроса возвращаемся к планированию
        graph.add_edge("ask_clarification_question", "plan_project_search")
        
        # После фильтрации проектов идем к ответу
        graph.add_conditional_edges(
            "filter_projects",
            self._route_after_filtering,
            {
                "found": "get_technical_answer",
                "not_found": "plan_project_search",  # Попробуем еще раз с новой информацией
            }
        )
        
        # Финальные узлы ведут к END
        graph.add_edge("get_technical_answer", END)
        graph.add_edge("get_organizational_answer", END)
        
        return graph.compile()
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Классифицирует запрос пользователя"""
        session_id = state.get("session_id", "unknown")
        user_query = state.get("user_query", "")
        
        logger.info(f"[Session: {session_id}] Классификация запроса: {user_query[:100]}")
        start_time = time.time()
        
        classification_prompt = load_prompt("classification.txt")
        full_prompt = f"{classification_prompt}\n\nЗапрос пользователя: {user_query}"
        
        query_type = self.llm.invoke_structured(
            full_prompt,
            response_format="одно слово: ТЕХНИЧЕСКИЙ или ОРГАНИЗАЦИОННЫЙ"
        ).strip()
        
        logger.debug(f"[Session: {session_id}] LLM ответ классификации: {query_type}")
        
        # Нормализуем ответ
        if "ТЕХНИЧЕСКИЙ" in query_type.upper():
            query_type = "ТЕХНИЧЕСКИЙ"
        elif "ОРГАНИЗАЦИОННЫЙ" in query_type.upper():
            query_type = "ОРГАНИЗАЦИОННЫЙ"
        else:
            # По умолчанию считаем техническим, если не понятно
            query_type = "ТЕХНИЧЕСКИЙ"
            logger.warning(f"[Session: {session_id}] Неопределенный тип запроса, установлен ТЕХНИЧЕСКИЙ")
        
        state["query_type"] = query_type
        elapsed = time.time() - start_time
        logger.info(f"[Session: {session_id}] Классификация завершена: {query_type} (время: {elapsed:.3f}s)")
        
        return state
    
    def _route_by_type(self, state: AgentState) -> str:
        """Маршрутизация по типу запроса"""
        return state["query_type"]
    
    def _plan_project_search(self, state: AgentState) -> AgentState:
        """Планирует поиск проекта: решает, достаточно ли информации или нужен вопрос"""
        session_id = state.get("session_id", "unknown")
        user_query = state.get("user_query", "")
        collected_info = state.get("collected_info", {})
        questions_asked = state.get("questions_asked", 0)
        
        logger.info(f"[Session: {session_id}] Планирование поиска проекта (вопросов задано: {questions_asked})")
        logger.debug(f"[Session: {session_id}] Запрос пользователя: {user_query}")
        logger.debug(f"[Session: {session_id}] Собранная информация: {collected_info}")
        
        start_time = time.time()
        
        # Получаем список всех проектов
        all_projects = self.db.get_all_projects()
        logger.debug(f"[Session: {session_id}] Загружено проектов для поиска: {len(all_projects)}")
        
        # Формируем список для промпта
        projects_list = "\n".join([
            f"- {proj['name']}: {proj['description']}"
            for proj in all_projects
        ])
        
        # Формируем строку с собранной информацией
        collected_info_str = ", ".join([
            f"{k}: {v}" for k, v in collected_info.items() if v
        ]) or "нет информации"
        
        # Загружаем промпт для планирования
        plan_prompt = load_prompt("plan_search.txt")
        formatted_prompt = format_prompt(
            plan_prompt,
            projects_list=projects_list,
            user_query=user_query,
            collected_info=collected_info_str
        )
        
        logger.debug(f"[Session: {session_id}] Отправка запроса к LLM для планирования")
        
        # Получаем ответ от LLM
        response = self.llm.invoke(formatted_prompt).strip()
        logger.debug(f"[Session: {session_id}] LLM ответ планирования (длина: {len(response)} символов)")
        
        # Парсим JSON ответ
        try:
            # Извлекаем JSON из ответа (может быть обернут в markdown)
            original_response = response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            plan = json.loads(response)
            logger.info(f"[Session: {session_id}] План распарсен: action={plan.get('action')}, reason={plan.get('reason', '')[:50]}")
        except (json.JSONDecodeError, ValueError) as e:
            # Если не удалось распарсить, используем дефолтные значения
            logger.error(f"[Session: {session_id}] Ошибка парсинга JSON плана: {e}")
            logger.debug(f"[Session: {session_id}] Ответ LLM: {original_response[:200]}")
            plan = {
                "action": "ask_question",
                "reason": "Не удалось распарсить ответ LLM",
                "question": "Уточните, пожалуйста, о каком проекте идет речь?",
                "question_type": "other"
            }
        
        # Сохраняем план в состояние
        state["search_plan"] = plan
        state["current_question"] = plan.get("question", "")
        state["needs_user_input"] = plan.get("action") == "ask_question"
        
        elapsed = time.time() - start_time
        action = plan.get("action", "unknown")
        logger.info(f"[Session: {session_id}] Планирование завершено: {action} (время: {elapsed:.3f}s)")
        
        if action == "ask_question":
            logger.info(f"[Session: {session_id}] Будет задан вопрос: {plan.get('question', '')[:100]}")
        elif action == "search":
            logger.info(f"[Session: {session_id}] Достаточно информации для поиска, начинаем фильтрацию")
        elif action == "insufficient_info":
            logger.warning(f"[Session: {session_id}] Недостаточно информации после всех уточнений")
        
        return state
    
    def _route_after_planning(self, state: AgentState) -> str:
        """Маршрутизация после планирования"""
        session_id = state.get("session_id", "unknown")
        plan = state.get("search_plan", {})
        action = plan.get("action", "ask_question")
        questions_asked = state.get("questions_asked", 0)
        
        logger.debug(f"[Session: {session_id}] Маршрутизация после планирования: action={action}, questions_asked={questions_asked}")
        
        # Проверяем лимит вопросов
        if action == "ask_question" and questions_asked >= 3:
            logger.warning(f"[Session: {session_id}] Достигнут лимит вопросов (3), переходим к insufficient_info")
            return "insufficient"
        
        logger.debug(f"[Session: {session_id}] Маршрутизация: {action}")
        return action
    
    def _ask_clarification_question(self, state: AgentState) -> AgentState:
        """Задает уточняющий вопрос пользователю"""
        session_id = state.get("session_id", "unknown")
        plan = state.get("search_plan", {})
        question = plan.get("question", "Уточните, пожалуйста, о каком проекте идет речь?")
        question_type = plan.get("question_type", "other")
        
        # Увеличиваем счетчик вопросов
        questions_asked = state.get("questions_asked", 0) + 1
        state["questions_asked"] = questions_asked
        
        logger.info(f"[Session: {session_id}] Задаем уточняющий вопрос #{questions_asked} (тип: {question_type})")
        logger.info(f"[Session: {session_id}] Вопрос: {question}")
        
        # Сохраняем вопрос в состояние
        state["current_question"] = question
        state["needs_user_input"] = True
        
        # Формируем ответ с вопросом
        state["answer"] = question
        
        return state
    
    def _filter_projects(self, state: AgentState) -> AgentState:
        """Фильтрует проекты на основе собранной информации"""
        session_id = state.get("session_id", "unknown")
        collected_info = state.get("collected_info", {})
        user_query = state.get("user_query", "")
        
        logger.info(f"[Session: {session_id}] Фильтрация проектов по собранной информации")
        logger.debug(f"[Session: {session_id}] Критерии фильтрации: {collected_info}")
        
        start_time = time.time()
        
        # Получаем список всех проектов
        all_projects = self.db.get_all_projects()
        logger.debug(f"[Session: {session_id}] Всего проектов для фильтрации: {len(all_projects)}")
        
        # Формируем список для промпта
        projects_list = "\n".join([
            f"- {proj['name']}: {proj['description']}"
            for proj in all_projects
        ])
        
        # Загружаем промпт для фильтрации
        filter_prompt = load_prompt("filter_projects.txt")
        formatted_prompt = format_prompt(
            filter_prompt,
            projects_list=projects_list,
            user_query=user_query,
            group=collected_info.get("group", ""),
            language=collected_info.get("language", ""),
            order=collected_info.get("order", ""),
            other_info=collected_info.get("other", "")
        )
        
        logger.debug(f"[Session: {session_id}] Отправка запроса к LLM для фильтрации проектов")
        
        # Получаем ответ от LLM
        response = self.llm.invoke(formatted_prompt).strip()
        logger.debug(f"[Session: {session_id}] LLM ответ фильтрации: {response[:200]}")
        
        # Парсим список проектов
        if "НЕИЗВЕСТНО" in response.upper():
            project_names = []
            logger.warning(f"[Session: {session_id}] LLM вернул НЕИЗВЕСТНО - проекты не найдены")
        else:
            # Разделяем по запятой и очищаем
            project_names = [
                name.strip() 
                for name in response.split(",")
                if name.strip()
            ]
            # Ограничиваем количество
            project_names = project_names[:MAX_PROJECT_SUGGESTIONS]
            logger.info(f"[Session: {session_id}] Найдено проектов: {len(project_names)} - {project_names}")
        
        state["project_names"] = project_names
        
        elapsed = time.time() - start_time
        logger.info(f"[Session: {session_id}] Фильтрация завершена: найдено {len(project_names)} проектов (время: {elapsed:.3f}s)")
        
        return state
    
    def _route_after_filtering(self, state: AgentState) -> str:
        """Маршрутизация после фильтрации проектов"""
        session_id = state.get("session_id", "unknown")
        project_names = state.get("project_names", [])
        questions_asked = state.get("questions_asked", 0)
        
        logger.debug(f"[Session: {session_id}] Маршрутизация после фильтрации: найдено {len(project_names)} проектов")
        
        if project_names:
            logger.info(f"[Session: {session_id}] Проекты найдены, переходим к формированию ответа")
            return "found"
        else:
            # Если проекты не найдены, проверяем, можно ли задать еще вопрос
            if questions_asked < 3:
                logger.info(f"[Session: {session_id}] Проекты не найдены, можно задать еще вопрос (задано: {questions_asked}/3)")
                return "not_found"  # Попробуем еще раз
            else:
                logger.warning(f"[Session: {session_id}] Проекты не найдены, достигнут лимит вопросов")
                return "found"  # Достигнут лимит, возвращаем пустой результат
    
    
    def _get_technical_answer(self, state: AgentState) -> AgentState:
        """Получает ответ на технический вопрос"""
        session_id = state.get("session_id", "unknown")
        user_query = state.get("user_query", "")
        project_names = state.get("project_names", [])
        needs_input = state.get("needs_user_input", False)
        
        logger.info(f"[Session: {session_id}] Формирование технического ответа")
        logger.debug(f"[Session: {session_id}] Запрос: {user_query}")
        logger.debug(f"[Session: {session_id}] Проекты: {project_names}")
        
        # Проверяем, есть ли вопрос для пользователя
        if needs_input:
            logger.debug(f"[Session: {session_id}] Нужен ответ от пользователя, возвращаем вопрос")
            return state
        
        # Проверяем, достигнут ли лимит вопросов без результата
        questions_asked = state.get("questions_asked", 0)
        plan = state.get("search_plan", {})
        
        if plan.get("action") == "insufficient" or (questions_asked >= 3 and not project_names):
            error_msg = (
                "К сожалению, не удалось найти проект по вашему запросу после всех уточнений. "
                "Попробуйте переформулировать запрос или указать полное название проекта."
            )
            logger.warning(f"[Session: {session_id}] Не удалось найти проект после {questions_asked} вопросов")
            state["answer"] = error_msg
            return state
        
        if not project_names:
            error_msg = "Не удалось определить проект по вашему запросу. Пожалуйста, уточните название проекта."
            logger.warning(f"[Session: {session_id}] Проекты не определены")
            state["answer"] = error_msg
            return state
        
        start_time = time.time()
        
        # Собираем контекст из всех найденных проектов
        project_contexts = []
        for project_name in project_names:
            logger.debug(f"[Session: {session_id}] Загрузка заданий для проекта: {project_name}")
            tasks = self.db.get_project_tasks(project_name)
            if tasks:
                project_contexts.append(f"## Проект: {project_name}\n\n{tasks}")
                logger.debug(f"[Session: {session_id}] Задания загружены для {project_name} (длина: {len(tasks)} символов)")
            else:
                logger.warning(f"[Session: {session_id}] Задания не найдены для проекта: {project_name}")
        
        if not project_contexts:
            error_msg = f"Не найдены задания для проектов: {', '.join(project_names)}"
            logger.error(f"[Session: {session_id}] {error_msg}")
            state["answer"] = error_msg
            return state
        
        # Объединяем контексты
        full_context = "\n\n".join(project_contexts)
        logger.debug(f"[Session: {session_id}] Общий контекст сформирован (длина: {len(full_context)} символов)")
        
        # Загружаем промпт для технического ответа
        technical_prompt = load_prompt("technical_answer.txt")
        formatted_prompt = format_prompt(
            technical_prompt,
            project_context=full_context,
            user_query=user_query
        )
        
        logger.debug(f"[Session: {session_id}] Отправка запроса к LLM для формирования ответа")
        
        # Получаем ответ от LLM
        answer = self.llm.invoke(formatted_prompt)
        logger.info(f"[Session: {session_id}] Ответ от LLM получен (длина: {len(answer)} символов)")
        
        state["answer"] = answer
        
        elapsed = time.time() - start_time
        logger.info(f"[Session: {session_id}] Технический ответ сформирован (время: {elapsed:.3f}s)")
        
        return state
    
    def _get_organizational_answer(self, state: AgentState) -> AgentState:
        """Получает ответ на организационный вопрос"""
        session_id = state.get("session_id", "unknown")
        user_query = state.get("user_query", "")
        
        logger.info(f"[Session: {session_id}] Формирование организационного ответа")
        logger.debug(f"[Session: {session_id}] Запрос: {user_query}")
        
        start_time = time.time()
        
        # Загружаем промпт для формирования поискового запроса
        search_prompt_template = load_prompt("rocketchat_search.txt")
        search_prompt = format_prompt(
            search_prompt_template,
            user_query=user_query
        )
        
        logger.debug(f"[Session: {session_id}] Отправка запроса к LLM для формирования поискового запроса RocketChat")
        
        # Получаем поисковый запрос от LLM
        search_query = self.llm.invoke(search_prompt).strip()
        logger.info(f"[Session: {session_id}] Поисковый запрос для RocketChat: {search_query}")
        
        state["search_query"] = search_query
        
        # Ищем в RocketChat
        logger.debug(f"[Session: {session_id}] Поиск в RocketChat")
        search_results = self.rocketchat.search(search_query)
        logger.info(f"[Session: {session_id}] Найдено результатов в RocketChat: {len(search_results)}")
        
        formatted_results = self.rocketchat.format_search_results(search_results)
        
        # Формируем финальный ответ
        answer = f"По вашему организационному вопросу:\n\n{formatted_results}"
        state["answer"] = answer
        
        elapsed = time.time() - start_time
        logger.info(f"[Session: {session_id}] Организационный ответ сформирован (время: {elapsed:.3f}s)")
        
        return state
    
    def invoke(self, user_query: str, user_response: str = None, session_state: dict = None, session_id: str = None) -> dict:
        """
        Обрабатывает запрос пользователя
        
        Args:
            user_query: Запрос пользователя (или ответ на уточняющий вопрос)
            user_response: Ответ пользователя на уточняющий вопрос (если есть)
            session_state: Состояние сессии для продолжения диалога
            
        Returns:
            Словарь с ответом и метаданными:
            {
                "answer": str - ответ агента,
                "needs_input": bool - нужен ли ответ от пользователя,
                "question": str - вопрос для пользователя (если needs_input=True),
                "session_state": dict - состояние для следующего запроса
            }
        """
        # Определяем session_id
        if not session_id:
            session_id = session_state.get("session_id") if session_state else f"session_{int(time.time() * 1000)}"
        
        logger.info(f"[Session: {session_id}] ===== Начало обработки запроса =====")
        logger.info(f"[Session: {session_id}] Запрос пользователя: {user_query[:200]}")
        if user_response:
            logger.info(f"[Session: {session_id}] Ответ пользователя на вопрос: {user_response}")
        
        start_time = time.time()
        
        # Восстанавливаем состояние сессии или создаем новое
        if session_state:
            logger.debug(f"[Session: {session_id}] Восстановление состояния сессии")
            initial_state: AgentState = {
                "user_query": session_state.get("original_query", user_query),
                "query_type": session_state.get("query_type"),
                "project_names": session_state.get("project_names", []),
                "project_context": session_state.get("project_context", ""),
                "search_query": session_state.get("search_query", ""),
                "answer": "",
                "collected_info": session_state.get("collected_info", {}),
                "questions_asked": session_state.get("questions_asked", 0),
                "current_question": "",
                "needs_user_input": False,
                "session_id": session_id,
            }
            
            # Если есть ответ пользователя, обновляем собранную информацию
            if user_response:
                plan = session_state.get("search_plan", {})
                question_type = plan.get("question_type", "other")
                collected_info = initial_state["collected_info"]
                
                logger.info(f"[Session: {session_id}] Обновление собранной информации: {question_type}={user_response}")
                
                # Сохраняем ответ в зависимости от типа вопроса
                if question_type == "group":
                    collected_info["group"] = user_response
                elif question_type == "language":
                    collected_info["language"] = user_response
                elif question_type == "order":
                    collected_info["order"] = user_response
                else:
                    collected_info["other"] = user_response
                
                initial_state["collected_info"] = collected_info
        else:
            # Новый запрос
            logger.debug(f"[Session: {session_id}] Создание нового состояния")
            initial_state: AgentState = {
                "user_query": user_query,
                "query_type": None,
                "project_names": [],
                "project_context": "",
                "search_query": "",
                "answer": "",
                "collected_info": {},
                "questions_asked": 0,
                "current_question": "",
                "needs_user_input": False,
                "session_id": session_id,
            }
        
        # Запускаем граф
        logger.debug(f"[Session: {session_id}] Запуск графа агента")
        final_state = self.graph.invoke(initial_state)
        
        # Формируем ответ
        elapsed = time.time() - start_time
        result = {
            "answer": final_state.get("answer", ""),
            "needs_input": final_state.get("needs_user_input", False),
            "question": final_state.get("current_question", ""),
            "session_state": {
                "original_query": final_state.get("user_query", user_query),
                "query_type": final_state.get("query_type"),
                "project_names": final_state.get("project_names", []),
                "project_context": final_state.get("project_context", ""),
                "search_query": final_state.get("search_query", ""),
                "collected_info": final_state.get("collected_info", {}),
                "questions_asked": final_state.get("questions_asked", 0),
                "search_plan": final_state.get("search_plan", {}),
                "session_id": session_id,
            }
        }
        
        logger.info(f"[Session: {session_id}] ===== Запрос обработан =====")
        logger.info(f"[Session: {session_id}] Тип запроса: {final_state.get('query_type')}")
        logger.info(f"[Session: {session_id}] Найдено проектов: {len(final_state.get('project_names', []))}")
        logger.info(f"[Session: {session_id}] Нужен ввод: {result['needs_input']}")
        logger.info(f"[Session: {session_id}] Вопросов задано: {final_state.get('questions_asked', 0)}")
        logger.info(f"[Session: {session_id}] Общее время обработки: {elapsed:.3f}s")
        
        return result
