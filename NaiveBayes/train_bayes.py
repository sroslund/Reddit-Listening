import bayes
import praw
from pmaw import PushshiftAPI

validate = True

agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

api = PushshiftAPI()

model = bayes.NaiveBayes()
#model.train(api, reddit)
#model.save('./trained_models/attempt2.csv')

model.load('./trained_models/attempt1.csv')

url = "https://www.reddit.com/r/UCSC/comments/11p9z11/whats_it_like_havingpursuing_a_crush_in_college/"
post = reddit.submission(url=url)
print(model.predict(post))


if validate:
    validation_set = {"politics" : ["https://www.reddit.com/r/UCSC/comments/10g1zrj/president_biden_is_coming_here_to_santa_cruz_on/",
                                    "https://www.reddit.com/r/UCSC/comments/jnpny1/hows_everyone_feeling_about_the_election/",
                                    "https://www.reddit.com/r/UCSC/comments/iw302q/only_read_if_youre_concerned_about_the_2020/",
                                    "https://www.reddit.com/r/UCSC/comments/y4zyop/fredo_has_always_been_like_this/",
                                    "https://www.reddit.com/r/UCSC/comments/uw11d2/college_republicans_im_shocked/",
                                    "https://www.reddit.com/r/UCSC/comments/f3lpfv/thoughts_video_titled_uc_santa_cruz_college/",
                                    "https://www.reddit.com/r/UCSC/comments/5pmo9k/does_ucsc_still_have_a_college_republicans_club/",
                                    "https://www.reddit.com/r/UCSC/comments/q1m51o/leftist_groups_and_clubs_on_campus/",
                                    "https://www.reddit.com/r/UCSC/comments/f6kk50/bernie_stands_with_the_strikers_king_shit/",
                                    "https://www.reddit.com/r/UCSC/comments/esgk39/just_an_incoherent_thought_after_listening_to_mlk/"],
                        "relationships" : ["https://www.reddit.com/r/UCSC/comments/11p9z11/whats_it_like_havingpursuing_a_crush_in_college/",
                                    "https://www.reddit.com/r/UCSC/comments/10vjger/if_your_friend_stinks_and_you_dont_tell_them/",
                                    "https://www.reddit.com/r/UCSC/comments/syaw7e/who_are_your_friends_at_ucsc/",
                                    "https://www.reddit.com/r/UCSC/comments/zwbwwo/making_friends/",
                                    "https://www.reddit.com/r/UCSC/comments/xn6ye2/any_girls_who_also_dont_have_any_friends_at_all/",
                                    "https://www.reddit.com/r/UCSC/comments/ygx5hr/where_do_u_find_parties/",
                                    "https://www.reddit.com/r/UCSC/comments/x69cml/how_is_the_party_scene_at_the_start_of_the_year/",
                                    "https://www.reddit.com/r/UCSC/comments/xwue5n/are_there_gonna_be_a_halloween_costume_party_the/",
                                    "https://www.reddit.com/r/UCSC/comments/10gks21/fun_clubs_or_groups_to_meet_friends/",
                                    "https://www.reddit.com/r/UCSC/comments/wp4pog/roommates/"],
                        "classwork" : ["https://www.reddit.com/r/UCSC/comments/11ctrm5/cse_12_sangnik_nath_or_marcelo_siero/",
                                    "https://www.reddit.com/r/UCSC/comments/10t3582/does_anyone_take_cse_107_with_professors_chen/",
                                    "https://www.reddit.com/r/UCSC/comments/zxluki/can_you_change_grade_from_pass_to_letter_grade_if/",
                                    "https://www.reddit.com/r/UCSC/comments/11c2fll/about_the_major_cs_game_design/",
                                    "https://www.reddit.com/r/UCSC/comments/11ezscp/would_it_be_a_good_idea_to_double_major_in/",
                                    "https://www.reddit.com/r/UCSC/comments/zw03uv/what_class_should_you_show_up_to_on_the_first_day/",
                                    "https://www.reddit.com/r/UCSC/comments/nltprm/thoughts_on_academic_integrity/",
                                    "https://www.reddit.com/r/UCSC/comments/rjhscx/is_there_a_way_to_do_19b_homework_early/",
                                    "https://www.reddit.com/r/UCSC/comments/uvrspw/school_policy_on_homework_exam_during_finals_week/",
                                    "https://www.reddit.com/r/UCSC/comments/fca42a/what_are_the_homework_assignments_in_cse138/"],
                        "jokes" : ["https://www.reddit.com/r/UCSC/comments/y8huwc/slugs_can_have_a_little_meme_as_a_treat/",
                                    "https://www.reddit.com/r/UCSC/comments/stm6ac/ucsc_going_viral_on_a_popular_meme_page/",
                                    "https://www.reddit.com/r/UCSC/comments/apz4ne/the_original_meme/",
                                    "https://www.reddit.com/r/UCSC/comments/10qagi4/i_know_you_go_to_ucsc_cuz_of_them_sexy_calves/",
                                    "https://www.reddit.com/r/UCSC/comments/d4e2pf/interesting_and_funny_title/",
                                    "https://www.reddit.com/r/UCSC/comments/tc5io9/the_successor_to_the_hello_kitty_porsche_is_the/",
                                    "https://www.reddit.com/r/UCSC/comments/w5gn0q/was_bored_so_i_made_a_ucsc_slander_video/",
                                    "https://www.reddit.com/r/UCSC/comments/droyos/cmon_give_it_a_go/",
                                    "https://www.reddit.com/r/UCSC/comments/fgg0tc/all_of_us_on_zoom_next_quarter/",
                                    "https://www.reddit.com/r/UCSC/comments/dmix5c/why_walk_when_you_could_take_the_journey_of_a/"],
                        "complaints" : ["https://www.reddit.com/r/UCSC/comments/11btt0v/enrollment_date_rant_ig/",
                                    "https://www.reddit.com/r/UCSC/comments/vbw3c1/landlords_are_the_worst/",
                                    "https://www.reddit.com/r/UCSC/comments/t9q39s/your_worst_housing_experiences/",
                                    "https://www.reddit.com/r/UCSC/comments/n0sh1f/my_worst_quarter/",
                                    "https://www.reddit.com/r/UCSC/comments/fibuvk/im_sad/",
                                    "https://www.reddit.com/r/UCSC/comments/et4wor/worst_class_experience_at_ucsc/",
                                    "https://www.reddit.com/r/UCSC/comments/bs4sy3/i_cant_help_but_be_angry/",
                                    "https://www.reddit.com/r/UCSC/comments/b22wnd/ucsc_roads_are_awful/",
                                    "https://www.reddit.com/r/UCSC/comments/yde816/im_so_close_to_giving_up/",
                                    "https://www.reddit.com/r/UCSC/comments/f67aor/loneliness_stress_depression/"],}
    
    for i in model.classes:
        correct_count = 0
        for j in validation_set[i]:
            post = reddit.submission(url=j)
            out = model.predict(post)
            print(out)
            if i in out:
                correct_count += 1
        print(f"{i}: {correct_count*10}% correct")