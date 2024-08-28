import dataclasses 

@dataclasses.dataclass
class ExampleData:

    example_data = [
        ("What is a panda bear?", "The giant panda is a bear species endemic to China.", "Yes"),
        ("What is a panda bear?", "Pandas are bears that like to eat bamboo.", "Yes"),
        ("What is a panda bear?", "There are bears in China that like to eat bamboo.", "No"),
        ("Where do pandas live?", "There are many bear species, like pandas, that live in the mountains of china.", "Yes"),
        ("What is a panda bear?", "Pandas is a powerful and open-source library Python library for data manipulation and analysis, providing data structures and functions for efficient operations.", "No"),
        ("Where do pandas live?", "In the pandas DataFrame library, you can use the `read_csv` function to import data from a CSV file.", "No")
    ]

    train_data = [
    # Positive Examples
        ("What is a panda?", "The giant panda is a bear species endemic to China.", "Yes"),
        ("What is the diet of a panda bear?", "In addition to bamboo, panda bears occasionally eat small mammals and carrion.", "Yes"),
        ("Where do panda bears live?", "Panda bears, also known as giant pandas, primarily live in the mountainous regions of central China, in Sichuan, Shaanxi, and Gansu provinces.", "Yes"),
        ("Do bears live in the jungle?", "The sloth bear is one species that lives in the tropical forests of India and Sri Lanka, making it an exception among bear species.", "Yes"),
        ("How long do bears hibernate?", "In warmer climates, some bears may only hibernate for a few weeks, while in colder regions, hibernation can last for several months.", "Yes"),
        ("How do bears find food?", "Bears have an exceptional sense of smell, which they use to locate food sources from miles away.", "Yes"),
        ("What time of year do bears give birth?", "Most bear species give birth during the winter months while hibernating, with cubs being born in the den.", "Yes"),
        ("Do all bears hibernate?", "While most bears hibernate, some species like the sun bear, which lives in tropical climates, do not enter true hibernation.", "Yes"),
        
        # Negative Examples
        ("What is the diet of a panda bear?", "Pandas DataFrame allows you to perform operations like `groupby` and `merge` for data manipulation.", "No"),
        ("Where do panda bears live?", "The `pivot_table` function in pandas allows for flexible reshaping of data.", "No"),
        ("What are the threats to panda bears?", "You can use the `dropna` function in pandas to remove missing values from a DataFrame.", "No"),
]


#     train_data = [
#     # Positive Examples
#     ("Where do polar bears live?", "Polar bears are most commonly found on sea ice in the Arctic Ocean, where they hunt seals.", "Yes"),
#     ("What is the diet of a panda bear?", "In addition to bamboo, panda bears occasionally eat small mammals and carrion.", "Yes"),
#     ("How do grizzly bears prepare for winter?", "Grizzly bears increase their caloric intake significantly during the late summer and fall to prepare for hibernation.", "Yes"),
#     ("What is the primary difference between black bears and brown bears?", "Brown bears typically have a large shoulder hump, which is a muscle used for digging, while black bears lack this feature.", "Yes"),
#     ("Do bears live in the jungle?", "The sloth bear is one species that lives in the tropical forests of India and Sri Lanka, making it an exception among bear species.", "Yes"),
#     ("How many hours of sleep do bears need?", "During hibernation, bears can go without waking for several months, depending on their species and the climate.", "Yes"),
#     ("What color is a sun bear?", "The sun bear's chest marking is unique to each individual and can vary in shape from crescent to a full circle.", "Yes"),
#     ("What are the threats to polar bear populations?", "Melting sea ice forces polar bears to travel greater distances for food, leading to exhaustion and starvation.", "Yes"),
#     ("How long do bears hibernate?", "In warmer climates, some bears may only hibernate for a few weeks, while in colder regions, hibernation can last for several months.", "Yes"),
#     ("What is the average weight of a male grizzly bear?", "Grizzly bears can weigh significantly more in the fall after they've built up fat reserves for hibernation.", "Yes"),
#     ("What are the differences between a polar bear and a grizzly bear?", "Polar bears have adapted to life in the Arctic with a thick layer of blubber and dense fur, while grizzly bears are more suited to forested and mountainous regions.", "Yes"),
#     ("How do bears communicate with each other?", "Bears use a combination of vocalizations, body language, and scent markings to communicate with one another.", "Yes"),
#     ("What role do bears play in their ecosystems?", "Bears help control prey populations and disperse seeds through their scat, playing a crucial role in maintaining ecosystem balance.", "Yes"),
#     ("How long can a bear live?", "In the wild, bears can live up to 25 years, though some individuals in protected environments have lived over 30 years.", "Yes"),
#     ("How fast can a bear run?", "Despite their size, bears can run at speeds up to 30-35 miles per hour, especially when charging.", "Yes"),
#     ("What types of bears are there?", "There are eight bear species: American black bear, brown bear, polar bear, Asiatic black bear, sloth bear, spectacled bear, sun bear, and giant panda.", "Yes"),
#     ("What are the physical characteristics of a black bear?", "Black bears have short, non-retractable claws and are typically smaller than brown bears, with a more streamlined build.", "Yes"),
#     ("How do bears find food?", "Bears have an exceptional sense of smell, which they use to locate food sources from miles away.", "Yes"),
#     ("What time of year do bears give birth?", "Most bear species give birth during the winter months while hibernating, with cubs being born in the den.", "Yes"),
#     ("Do all bears hibernate?", "While most bears hibernate, some species like the sun bear, which lives in tropical climates, do not enter true hibernation.", "Yes"),

#     # Negative Examples
#     ("Where do polar bears live?", "Albert Einstein developed the theory of relativity.", "No"),
#     ("What is the diet of a panda bear?", "Mount Everest is the highest peak in the world, standing at 8,848 meters.", "No"),
#     ("How do grizzly bears prepare for winter?", "The Colosseum in Rome is an ancient amphitheater and a symbol of the Roman Empire.", "No"),
#     ("What is the primary difference between black bears and brown bears?", "The Great Barrier Reef is the largest coral reef system in the world, located off the coast of Australia.", "No"),
#     ("Do bears live in the jungle?", "Venus is the second planet from the sun and is known for its extreme temperatures.", "No"),
#     ("How many hours of sleep do bears need?", "The Nile River is the longest river in the world, flowing through northeastern Africa.", "No"),
#     ("What color is a sun bear?", "The Leaning Tower of Pisa is famous for its unintended tilt.", "No"),
#     ("What are the threats to polar bear populations?", "The human respiratory system is responsible for taking in oxygen and expelling carbon dioxide.", "No"),
#     ("How long do bears hibernate?", "Walt Disney founded the Disney Company, which became a global leader in entertainment.", "No"),
#     ("What is the average weight of a male grizzly bear?", "The Pacific Ocean is the largest and deepest of the world's oceanic divisions.", "No"),
#     ("Where do polar bears live?", "The Taj Mahal is an ivory-white marble mausoleum located in Agra, India.", "No"),
#     ("What is the diet of a panda bear?", "The Amazon Rainforest is known for its biodiversity and is often referred to as the 'lungs of the Earth'.", "No"),
#     ("How do grizzly bears prepare for winter?", "The Statue of Liberty was a gift from France to the United States and is located in New York Harbor.", "No"),
#     ("What is the primary difference between black bears and brown bears?", "Marie Curie was the first woman to win a Nobel Prize and is famous for her research on radioactivity.", "No"),
#     ("Do bears live in the jungle?", "The Louvre Museum in Paris is the world's largest art museum and a historic monument.", "No"),
#     ("How many hours of sleep do bears need?", "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate Strait in California.", "No"),
#     ("What color is a sun bear?", "The Sahara Desert stretches across North Africa and is known for its harsh, arid climate.", "No"),
#     ("What are the threats to polar bear populations?", "The invention of the printing press by Johannes Gutenberg revolutionized the dissemination of information.", "No"),
#     ("How long do bears hibernate?", "The moon is Earth's only natural satellite and has a significant influence on the planet's tides.", "No"),
#     ("What is the average weight of a male grizzly bear?", "The Pyramids of Giza are ancient pyramid structures built by the Egyptians.", "No"),
# ]