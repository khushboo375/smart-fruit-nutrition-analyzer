from google import genai

client = genai.Client(api_key="your_api_key")


def generate_ai_recommendation(
    fruit_counts,
    total_calories,
    total_carbs,
    total_protein,
    total_fat,
    daily_calories,
    bmi=None
):
    try:
        prompt = f"""
        Fruits detected: {fruit_counts}

        Total Calories: {total_calories}
        Carbohydrates: {total_carbs}
        Protein: {total_protein}
        Fat: {total_fat}
        Daily Limit: {daily_calories}

        BMI: {bmi if bmi else "Not provided"}

        Give a short and simple health recommendation.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        print(" Gemini API failed:", e)

        #  fallback call
        return generate_fallback_recommendation(
            fruit_counts,
            {
                "calories": total_calories,
                "carbs": total_carbs,
                "protein": total_protein,
                "fat": total_fat
            },
            bmi,
            daily_calories
        )
    
    
def generate_fallback_recommendation(fruit_counts, nutrition_data, bmi=None, daily_calories=2000, goal="maintenance"):

    total_calories = nutrition_data.get("calories", 0)
    total_carbs = nutrition_data.get("carbs", 0)
    total_protein = nutrition_data.get("protein", 0)
    total_fat = nutrition_data.get("fat", 0)
    total_fiber = nutrition_data.get("fiber", 0)

    #  SCORE SYSTEM
    score = 100

    if total_protein < 10:
        score -= 15
    if total_fiber < 5:
        score -= 15
    if total_carbs > 60:
        score -= 10
    if total_calories > daily_calories:
        score -= 10

    score = max(score, 50)

    #  ANALYSIS
    analysis = []

    if total_calories > daily_calories:
        analysis.append("You are consuming more calories than required.")
    elif total_calories < daily_calories * 0.5:
        analysis.append("Your calorie intake is too low.")
    else:
        analysis.append("Your calorie intake is well balanced.")

    if total_protein < 10:
        analysis.append("Protein intake is insufficient.")
    if total_fiber < 5:
        analysis.append("Fiber intake is low.")
    if total_carbs > 60:
        analysis.append("Carbohydrates are slightly high.")

    #  RISKS
    risks = []

    if total_carbs > 60:
        risks.append("May lead to sugar spikes.")
    if total_fiber < 5:
        risks.append("May affect digestion.")
    if total_protein < 10:
        risks.append("May impact muscle health.")
    if total_calories > daily_calories:
        risks.append("Risk of weight gain.")

    if not risks:
        risks.append("No major risks detected.")

    #  GOAL-BASED ADVICE
    goal_advice = []

    if goal == "weight_loss":
        goal_advice.append("Focus on calorie deficit and high-fiber foods.")
    elif goal == "weight_gain":
        goal_advice.append("Increase calorie intake with protein-rich foods.")
    else:
        goal_advice.append("Maintain a balanced diet.")

    # SMART FOOD SWAPS
    swaps = []

    if total_carbs > 60:
        swaps.append("Replace high-sugar fruits with apples or berries.")
    if total_protein < 10:
        swaps.append("Add paneer, eggs, or lentils.")
    if total_fiber < 5:
        swaps.append("Include oats, broccoli, or whole fruits.")

    #  HABITS
    habits = [
        "Drink at least 2-3 liters of water daily.",
        "Avoid late-night heavy meals.",
        "Include a balanced mix of carbs, protein, and fiber."
    ]

    #  BMI
    bmi_section = ""
    if bmi:
        if bmi < 18.5:
            status = "Underweight"
        elif bmi < 24.9:
            status = "Normal"
        elif bmi < 29.9:
            status = "Overweight"
        else:
            status = "Obese"

        bmi_section = f"Your BMI is {bmi:.1f} ({status})."

    # FINAL OUTPUT 
    result = f"""
Health Score: {score}/100

Analysis:
- {'\n- '.join(analysis)}

Risks:
- {'\n- '.join(risks)}

Goal Guidance:
- {'\n- '.join(goal_advice)}

Smart Swaps:
- {'\n- '.join(swaps) if swaps else "No swaps needed"}

Healthy Habits:
- {'\n- '.join(habits)}

BMI:
{bmi_section if bmi_section else "Not provided"}
"""

    return result.strip()