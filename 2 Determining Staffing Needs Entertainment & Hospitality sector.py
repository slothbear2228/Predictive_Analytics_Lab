import math

def calculate_staffing_needs(expected_guests, avg_service_time_min, operating_hours, guests_per_staff_per_hour):
    """
    Calculate number of staff needed.

    Parameters:
    - expected_guests: int, number of guests expected
    - avg_service_time_min: float, average service time per guest (minutes)
    - operating_hours: float, total hours of operation
    - guests_per_staff_per_hour: int, guests one staff can serve per hour

    Returns:
    - int: number of staff needed (rounded up)
    """
    total_service_time_hours = (expected_guests * avg_service_time_min) / 60
    staff_needed = total_service_time_hours / operating_hours / (guests_per_staff_per_hour / 1)
    return math.ceil(staff_needed)

def main():
    print("Staffing Needs Calculator for Entertainment & Hospitality Sector")

    expected_guests = int(input("Enter expected number of guests: "))
    avg_service_time_min = float(input("Enter average service time per guest (in minutes): "))
    operating_hours = float(input("Enter operating hours: "))
    guests_per_staff_per_hour = int(input("Enter how many guests one staff member can handle per hour: "))

    staff_required = calculate_staffing_needs(expected_guests, avg_service_time_min, operating_hours, guests_per_staff_per_hour)

    print(f"\nEstimated number of staff required: {staff_required}")

if __name__ == "__main__":
    main()
