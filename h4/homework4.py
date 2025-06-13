import random
import requests
import folium


class Point:
    def __init__(self, name, lat, lng, weight):
        self.name = name
        self.lat = lat
        self.lng = lng
        self.weight = weight


class RouteOptimizer:
    def __init__(self, points, max_time_minutes, transport_type="driving"):
        print("Инициализация RouteOptimizer")
        self.points = points
        self.max_time_minutes = max_time_minutes
        # Размер популяции
        self.population_size = 50
        # Количество поколений
        self.generations = 100
        # Вероятность мутации
        self.mutation_rate = 0.2
        # Количество элитных особей
        self.elite_size = 5
        self.transport_type = transport_type
        # OSRM сервер
        self.osrm_url = f"https://router.project-osrm.org/route/v1/{transport_type}"
        # Кэш
        self.route_cache = {}
        self.speed_coefficients = {
            "driving": 1.0,  # Стандарт
            "walking": 3.0,  # Медленнее в 3р
            "cycling": 1.5  # Медленнее в 1.5р
        }

        print(f"OSRM: {self.osrm_url}")
        print(f"Тип передвижения: {transport_type}")
        print(f"Коэффициент скорости: {self.speed_coefficients[transport_type]}x")

    # Функция для подсчета времени и веса пути
    def calculate_route_time(self, route):
        print(f"Расчет времени для маршрута с {len(route)} точками")
        total_time = 0
        total_weight = 0
        segment_times = []

        for i in range(len(route) - 1):

            # Формируем ключ для кэша
            key = (route[i].lng, route[i].lat, route[i + 1].lng, route[i + 1].lat)

            # Смотрим в кэш
            if key in self.route_cache:
                duration = self.route_cache[key]
                print(f"CACHE: Время между точками {i} и {i + 1}: {duration:.2f} минут")
            else:
                coordinates = f"{route[i].lng},{route[i].lat};{route[i + 1].lng},{route[i + 1].lat}"

                url = f"{self.osrm_url}/{coordinates}"
                params = {
                    "overview": "false",
                    "alternatives": "false"
                }

                print(f"REQUEST: {url}")
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    # Переводим время в минуты
                    duration = data['routes'][0]['duration'] / 60
                    # Учитываем коэффициент
                    duration *= self.speed_coefficients[self.transport_type]
                    # Кладём в кэш
                    self.route_cache[key] = duration
                    print(f"Время между точками {i} и {i + 1}: {duration:.2f} минут")

            segment_times.append(duration)
            total_time += duration
            total_weight += route[i].weight

        # Добавляем вес последней точки
        total_weight += route[-1].weight

        print("\nМетрики маршрута:")
        for i, time in enumerate(segment_times):
            print(f"Сегмент {i + 1}: {time:.2f} минут")
        print(f"Общее время маршрута: {total_time:.2f} минут")
        print(f"Общий вес маршрута: {total_weight:.2f}")

        return total_time, total_weight

    # Считаем фитнес маршрута (чем больше число на выходе -> тем лучше маршрут)
    def fitness(self, route):
        total_time, total_weight = self.calculate_route_time(route)

        # Штраф: Если превысили лимит по времени
        if total_time > self.max_time_minutes:
            return 0

        # Штраф: Если точки не идут в порядке убывания веса
        weight_penalty = 0
        for i in range(len(route) - 1):
            if route[i].weight < route[i + 1].weight:
                weight_penalty += (route[i + 1].weight - route[i].weight) * 2

        # Возвращаем общий вес с учетом штрафа
        return total_weight - weight_penalty
        # Создаём начальную популяцию
    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            route = self.points.copy()
            random.shuffle(route)
            population.append(route)
        return population

    # Скрещиваем двух родителей
    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        # Создаем детей родителя 1
        child = [None] * size
        child[start:end] = parent1[start:end]

        # Оставшиеся позиции заполняем элементами родителя 2
        remaining = [x for x in parent2 if x not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1

        return child

    # Мутация
    def mutate(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # Выбор родителя
    def select_parent(self, population):
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness)

    # Оптимизация маршрута
    def optimize(self):
        population = self.create_initial_population()

        for generation in range(self.generations):
            new_population = []

            # Элитарность: сохраняем лучшее решение
            best_route = min(population, key=self.fitness)
            new_population.append(best_route)

            # Создаем новые популяции
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

            if (generation + 1) % 10 == 0:
                best_fitness = self.fitness(best_route)
                print(f"Поколение {generation + 1}, Лучшая приспособленность: {best_fitness}")

        return max(population, key=self.fitness)

    # Генерим HTML
    def visualize_route(self, route):
        m = folium.Map(location=[route[0].lat, route[0].lng], zoom_start=12)

        # Ставим точки на карту
        for point in route:
            folium.Marker(
                [point.lat, point.lng],
                popup=f"{point.name} (Вес: {point.weight})"
            ).add_to(m)

        # Получаем геометрию маршрута
        coordinates = ";".join([f"{point.lng},{point.lat}" for point in route])
        url = f"{self.osrm_url}/{coordinates}"
        params = {
            "overview": "full",
            "geometries": "geojson"
        }

        print(f"Запрос геометрии маршрута: {url}")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            route_geometry = data['routes'][0]['geometry']['coordinates']
            folium.PolyLine(
                [[coord[1], coord[0]] for coord in route_geometry],
                weight=2,
                color='blue',
                opacity=0.8
            ).add_to(m)

        m.save("kazan_route.html")
        print("Карта создана")


def main():
    print("-----------START-----------")

    points = [
        Point("Спасская Башня", 55.796514, 49.10839, 7),
        Point("Двойка", 55.792139, 49.122135, 6),
        Point("DDX", 55.786100, 49.121479, 5),
        Point("Театр Камала", 55.786100, 49.121479, 4),
        Point("Казанский цирк", 55.798770, 49.100534, 3),
        Point("Центр семьи Казан", 55.812725, 49.108306, 2),
        Point("Аквапарк Ривьера", 55.815406, 49.132332, 1),
    ]

    # Тип передвижения: driving, walking, cycling
    optimizer = RouteOptimizer(points, max_time_minutes=360, transport_type="walking")

    print("Поиск оптимального маршрута")
    best_route = optimizer.optimize()
        # Рассчитываем детали финального маршрута
    total_time, total_weight = optimizer.calculate_route_time(best_route)

    print(f"\nНайден лучший маршрут по Казани:")
    print(f"Тип передвижения: {optimizer.transport_type}")
    print(f"Общее время: {total_time:.2f} минут")
    print(f"Общий вес (приоритет): {total_weight:.2f}")
    print("\nПорядок посещения:")
    for i, point in enumerate(best_route, 1):
        print(f"{i}. {point.name} (приоритет: {point.weight})")

    optimizer.visualize_route(best_route)
    print("-----------END-----------")


if __name__ == "__main__":
    main()