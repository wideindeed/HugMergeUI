"""Domain-specific held-out passages, mirroring eval_texts.py's design but
targeted at math and code competency specifically. Generic-prose perplexity
(eval_texts.py) can stay flat even when a merge quietly degrades a
fine-tune's actual specialty - a math+code merge can still sound fluent
describing coral reefs while having lost math/code ability. These sets let
perplexity actually notice that kind of domain-specific damage.
"""

MATH_EVAL_TEXTS = [
    """
    To solve the quadratic equation x^2 - 5x + 6 = 0 using the quadratic
    formula, first identify the coefficients a = 1, b = -5, and c = 6.
    Substituting into x = (-b +/- sqrt(b^2 - 4ac)) / 2a gives
    x = (5 +/- sqrt(25 - 24)) / 2 = (5 +/- 1) / 2, yielding the two roots
    x = 3 and x = 2. Checking by substitution confirms both values satisfy
    the original equation.
    """.strip(),
    """
    Integration by parts follows from the product rule and states that the
    integral of u dv equals uv minus the integral of v du. To evaluate the
    integral of x times e^x dx, let u = x and dv = e^x dx, so du = dx and
    v = e^x. This gives x e^x minus the integral of e^x dx, which simplifies
    to x e^x minus e^x plus a constant of integration.
    """.strip(),
    """
    A train leaves station A traveling at 60 miles per hour at the same
    moment a second train leaves station B, 300 miles away, traveling toward
    the first train at 40 miles per hour. Since the trains close the
    distance at a combined rate of 100 miles per hour, they will meet after
    300 divided by 100, or 3 hours. At that point the first train has
    covered 180 miles and the second has covered 120 miles.
    """.strip(),
    """
    Bayes' theorem relates conditional probabilities: P(A|B) equals
    P(B|A) times P(A), divided by P(B). Suppose a disease affects 1% of a
    population, and a test for it is 95% accurate for both true positives
    and true negatives. Given a positive test result, the probability the
    person actually has the disease is P(B|A)P(A) divided by
    [P(B|A)P(A) + P(B|not A)P(not A)], which works out to roughly 16%,
    illustrating how a low base rate keeps the posterior probability low
    even with a fairly accurate test.
    """.strip(),
    """
    To prove by induction that the sum of the first n positive integers
    equals n(n+1)/2, first verify the base case n = 1, where the sum is 1
    and the formula gives 1(2)/2 = 1. Next assume the formula holds for some
    k, so that 1 + 2 + ... + k = k(k+1)/2. Adding k+1 to both sides gives
    k(k+1)/2 + (k+1), which factors to (k+1)(k+2)/2, matching the formula
    for n = k+1 and completing the inductive step.
    """.strip(),
]

CODE_EVAL_TEXTS = [
    """
    def binary_search(arr, target):
        \"\"\"Return the index of target in sorted arr, or -1 if absent.\"\"\"
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
    """.strip(),
    """
    def quicksort(arr):
        \"\"\"Sort a list in ascending order using the quicksort algorithm.\"\"\"
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """.strip(),
    """
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    users = {}

    @app.route("/users/<int:user_id>", methods=["GET"])
    def get_user(user_id):
        user = users.get(user_id)
        if user is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(user)

    @app.route("/users", methods=["POST"])
    def create_user():
        data = request.get_json()
        user_id = len(users) + 1
        users[user_id] = data
        return jsonify({"id": user_id}), 201
    """.strip(),
    """
    def fibonacci(n, memo=None):
        \"\"\"Return the nth Fibonacci number using memoized recursion.\"\"\"
        if memo is None:
            memo = {}
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
        return memo[n]
    """.strip(),
    """
    class LinkedList:
        class Node:
            def __init__(self, value, next=None):
                self.value = value
                self.next = next

        def __init__(self):
            self.head = None

        def insert_front(self, value):
            self.head = self.Node(value, self.head)

        def delete(self, value):
            prev, curr = None, self.head
            while curr is not None:
                if curr.value == value:
                    if prev is None:
                        self.head = curr.next
                    else:
                        prev.next = curr.next
                    return True
                prev, curr = curr, curr.next
            return False
    """.strip(),
]
