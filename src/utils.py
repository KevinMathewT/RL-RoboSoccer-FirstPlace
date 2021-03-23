import matplotlib.pyplot as plt
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util
from IPython import display

# to render in notebook
def colab_render(env, model):
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(ob)
        # print(action)
        ob, reward, done, info = env.step(action)

        plt.clf()
        title_str = ("reward : " + str(reward))
        padding = 5
        ax = plt.axes(xlim=(0 - padding, env.env.width + padding),
                      ylim=(0 - padding, env.env.height + padding))
        ax.set_aspect("equal")
        o = pymunk.matplotlib_util.DrawOptions(ax)
        env.env.space.debug_draw(o)
        plt.title(title_str, loc='left')
        display.display(plt.gcf())
        display.clear_output(wait=True)

        total_reward += reward
    return total_reward


def get_score(env, model):
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(ob)
        # print(action)
        ob, reward, done, info = env.step(action)

        total_reward += reward
    return total_reward

# to render in a popup-application
def game_render(env, model):
    ob = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done:
        step += 1
        action, _states = model.predict(ob)
        # print(action)
        ob, reward, done, info = env.step(action)

        # plt.figure(3)
        # plt.clf()
        # plt.imshow(env.render())
        # plt.title("%s. Step: %d" % (env._spec.id, step))

        # plt.pause(0.001)  # pause for plots to update

        plt.clf()
        title_str = ("reward : " + str(reward))
        padding = 5
        ax = plt.axes(xlim=(0 - padding, env.env.width + padding),
                      ylim=(0 - padding, env.env.height + padding))
        ax.set_aspect("equal")
        o = pymunk.matplotlib_util.DrawOptions(ax)
        env.env.space.debug_draw(o)
        plt.title(title_str, loc='left')
        plt.show(block=False)
        plt.pause(0.001)  # pause for plots to update

        total_reward += reward
    return total_reward
